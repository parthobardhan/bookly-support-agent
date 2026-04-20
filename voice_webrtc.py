"""
streamlit-webrtc AudioProcessor: mic frames -> Flux (linear16 mono 16 kHz).
"""

from __future__ import annotations

from typing import List

import av
from streamlit_webrtc import AudioProcessorBase

from audio_handler import FluxBridge, audioframe_to_flux_pcm


class FluxAudioProcessor(AudioProcessorBase):
    """Passes resampled PCM to a FluxBridge (started on first frame)."""

    def __init__(self, bridge: FluxBridge) -> None:
        super().__init__()
        self._bridge = bridge

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self._bridge.start()
        pcm = audioframe_to_flux_pcm(frame)
        if pcm:
            self._bridge.push_pcm(pcm)
        return frame

    def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        """Process every frame in the batch (avoids drops in async mode)."""
        self._bridge.start()
        for frame in frames:
            pcm = audioframe_to_flux_pcm(frame)
            if pcm:
                self._bridge.push_pcm(pcm)
        return frames
