"""
Voice: Deepgram Nova (batch STT), Flux (streaming STT + EOT), Aura REST / Speak WS (TTS).

Set DEEPGRAM_API_KEY. Flux path uses Listen v2; streaming TTS uses Speak v1 WebSocket (linear16 → WAV).
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
import wave
from typing import Optional

import httpx

import config

DEEPGRAM_BASE = "https://api.deepgram.com"


def _eot_normalize_key(transcript: str) -> str:
    return " ".join((transcript or "").lower().split())


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """
    Transcribe raw audio bytes using Deepgram Nova (pre-recorded listen API).
    """
    if not audio_bytes:
        return ""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    url = f"{DEEPGRAM_BASE}/v1/listen"
    params = {
        "model": config.DEEPGRAM_STT_MODEL,
        "smart_format": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mime_type,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, params=params, content=audio_bytes, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    try:
        return (
            data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Deepgram STT response shape: {data!r}") from exc


def generate_speech(
    text: str,
    *,
    model: Optional[str] = None,
) -> tuple[bytes, str]:
    """
    Synthesize speech with Deepgram Aura (single-request speak API).

    Returns (audio_bytes, mime_type) suitable for st.audio.
    """
    if not text.strip():
        return b"", "audio/mpeg"

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    url = f"{DEEPGRAM_BASE}/v1/speak"
    params = {"model": model or config.DEEPGRAM_TTS_MODEL}
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {"text": text.strip()}

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, params=params, json=payload, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "audio/mpeg")
        return resp.content, content_type


def pcm_s16le_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def synthesize_speech_wav(
    text: str,
    *,
    model: Optional[str] = None,
    sample_rate: Optional[int] = None,
) -> bytes:
    """
    Streaming TTS via Deepgram Speak v1 WebSocket: collect linear16 chunks, return WAV bytes.
    """
    if not text.strip():
        return b""

    from deepgram import DeepgramClient
    from deepgram.speak.v1.types.speak_v1flushed import SpeakV1Flushed
    from deepgram.speak.v1.types.speak_v1text import SpeakV1Text

    sr = sample_rate or config.DEEPGRAM_SPEAK_SAMPLE_RATE
    client = DeepgramClient()
    sr_str = str(sr)
    pcm = bytearray()
    with client.speak.v1.connect(
        model=model or config.DEEPGRAM_TTS_MODEL,
        encoding="linear16",
        sample_rate=sr_str,  # type: ignore[arg-type]
    ) as speak:
        speak.send_text(SpeakV1Text(type="Speak", text=text.strip()))
        speak.send_flush()
        while True:
            msg = speak.recv()
            if isinstance(msg, bytes):
                pcm.extend(msg)
            elif isinstance(msg, SpeakV1Flushed):
                break
        speak.send_close()
    return pcm_s16le_to_wav(bytes(pcm), sr, channels=1)


class FluxBridge:
    """
    Streams PCM (linear16 mono 16kHz) to Deepgram Flux; on EndOfTurn pushes final transcript to queue.
    """

    _global_mute_until = 0.0
    _global_mute_lock = threading.Lock()

    @classmethod
    def set_global_mute_until(cls, monotonic_until: float) -> None:
        """Mute even when no bridge is in session (e.g. TTS after WebRTC stopped)."""
        with cls._global_mute_lock:
            cls._global_mute_until = max(time.monotonic(), monotonic_until)

    def _effective_mute_until(self) -> float:
        with self._mute_lock:
            inst = self._input_mute_until
        with FluxBridge._global_mute_lock:
            g = FluxBridge._global_mute_until
        return max(inst, g)

    def __init__(self, transcript_queue: "queue.Queue[str]") -> None:
        self._transcript_queue = transcript_queue
        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._stop = threading.Event()
        self._conn = None
        self._ctx = None
        self._client = None
        self._listen_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()
        self._mute_lock = threading.Lock()
        self._input_mute_until = 0.0
        self._eot_dedupe_lock = threading.Lock()
        self._last_eot_norm_key = ""
        self._last_eot_monotonic = 0.0

    @property
    def is_active(self) -> bool:
        return self._started

    def _on_message(self, msg: object) -> None:
        from deepgram.listen.v2.types.listen_v2turn_info import ListenV2TurnInfo

        if isinstance(msg, ListenV2TurnInfo) and msg.event == "EndOfTurn":
            t = (msg.transcript or "").strip()
            if t:
                if time.monotonic() < self._effective_mute_until():
                    return
                nk = _eot_normalize_key(t)
                with self._eot_dedupe_lock:
                    now_m = time.monotonic()
                    if (
                        nk
                        and nk == self._last_eot_norm_key
                        and (now_m - self._last_eot_monotonic) < 15.0
                    ):
                        return
                    self._last_eot_norm_key = nk
                    self._last_eot_monotonic = now_m
                self._transcript_queue.put(t)

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if time.monotonic() < self._effective_mute_until():
                continue
            if self._conn is not None:
                try:
                    self._conn.send_media(chunk)
                except Exception:
                    break

    def _start_listening(self) -> None:
        if self._conn is not None:
            self._conn.start_listening()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop.clear()

            from deepgram import DeepgramClient
            from deepgram.core.events import EventType

            kwargs = {
                "model": config.DEEPGRAM_FLUX_MODEL,
                "encoding": "linear16",
                "sample_rate": 16000,
                "eot_threshold": config.FLUX_EOT_THRESHOLD,
                "eot_timeout_ms": config.FLUX_EOT_TIMEOUT_MS,
            }
            if config.FLUX_EAGER_EOT_THRESHOLD:
                kwargs["eager_eot_threshold"] = config.FLUX_EAGER_EOT_THRESHOLD

            self._client = DeepgramClient()
            self._ctx = self._client.listen.v2.connect(**kwargs)
            self._conn = self._ctx.__enter__()
            self._conn.on(EventType.MESSAGE, self._on_message)
            self._listen_thread = threading.Thread(target=self._start_listening, daemon=True)
            self._listen_thread.start()
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()

    def mute_input_for_seconds(self, seconds: float) -> None:
        """Drop mic PCM for a window so speaker TTS is not re-transcribed (acoustic loop)."""
        with self._mute_lock:
            until = time.monotonic() + max(0.0, seconds)
            self._input_mute_until = max(self._input_mute_until, until)

    def replace_mute_until(self, monotonic_until: float) -> None:
        """Set absolute mute end (e.g. after measuring WAV length)."""
        with self._mute_lock:
            self._input_mute_until = max(time.monotonic(), monotonic_until)

    def push_pcm(self, chunk: bytes) -> None:
        if chunk and self._started:
            if time.monotonic() < self._effective_mute_until():
                return
            self._audio_queue.put(chunk)

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            with self._mute_lock:
                self._input_mute_until = 0.0
            self._stop.set()
            self._audio_queue.put(None)
            if self._conn is not None:
                try:
                    self._conn.send_close_stream()
                except Exception:
                    pass
            if self._ctx is not None and self._conn is not None:
                try:
                    self._ctx.__exit__(None, None, None)
                except Exception:
                    pass
            self._conn = None
            self._ctx = None
            self._client = None
            self._started = False
            self._listen_thread = None
            self._send_thread = None
            with self._eot_dedupe_lock:
                self._last_eot_norm_key = ""
                self._last_eot_monotonic = 0.0


def audioframe_to_flux_pcm(frame) -> bytes:
    """Resample/reformat av.AudioFrame to linear16 mono 16kHz PCM bytes."""
    import av

    if not isinstance(frame, av.AudioFrame):
        return b""
    f = frame.reformat(format="s16", layout="mono", rate=16000)
    return bytes(f.planes[0])
