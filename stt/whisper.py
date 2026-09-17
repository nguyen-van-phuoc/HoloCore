import queue
import time
from typing import Optional
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel


class WhisperRecognizer:
    def __init__(
        self,
        model_size: str = "small",
        language: Optional[str] = None,       # None = automatically recognize
        device: Optional[str] = None,         # None = Select CUDA/CPU
        compute_type: Optional[str] = None,   # None = float16/int8 according to device
        sample_rate: int = 16000,             # Whisper requires 16kHz.
        block_ms: int = 30,                   # audio block size (30ms)
        silence_duration: float = 2.0,        # 2-second pause -> sentence break
        silence_threshold: float = 0.012,     # RMS threshold considered silence
        min_speech_duration: float = 0.3,     # Skip sentences that are too short.
        max_speech_duration: float = 30.0,    # Mandatory cut-off if the speech runs too long.
        start_timeout: Optional[float] = None,  # None = waiting to speak endlessly
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        # --- Select device ---
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"[STT] Loading Whisper '{model_size}' ON {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] The model is ready.")

        # --- Config ---
        self.language = language
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.start_timeout = start_timeout
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 audio array at self.sample_rate."""
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def listen(self, start_timeout: Optional[float] = None) -> Optional[str]:
        """
        Blocking: opens the microphone, records an utterance until a silence is detected,
        transcribes it, and returns the text.

        Returns:
            - str   : the recognized content
            - None  : no speech detected (start_timeout expired) or the utterance was too short
        """
        if start_timeout is None:
            start_timeout = self.start_timeout

        block_sec = self.block_size / self.sample_rate
        silence_blocks_needed = max(1, int(self.silence_duration / block_sec))
        max_blocks = max(1, int(self.max_speech_duration / block_sec))

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _cb(indata, frames, time_info, status):
            if status:
                print(f"[STT] audio status: {status}")
            audio_q.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=_cb,
        )

        buffer: list = []
        silence_count = 0
        speaking = False
        t_start = time.monotonic()

        try:
            stream.start()
            while True:
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    # It only times out if speaking hasn't started yet.
                    if (not speaking and start_timeout is not None
                            and time.monotonic() - t_start > start_timeout):
                        return None
                    continue

                rms = float(np.sqrt(np.mean(block ** 2)))
                is_speech = rms >= self.silence_threshold

                if is_speech:
                    speaking = True
                    silence_count = 0
                    buffer.append(block)
                elif speaking:
                    buffer.append(block)
                    silence_count += 1
                    if silence_count >= silence_blocks_needed:
                        break  # sufficient silence -> end of sentence

                # too long -> cut
                if speaking and len(buffer) >= max_blocks:
                    break

                # Before anything was said, the start_timeout was already over -> discard.
                if (not speaking and start_timeout is not None
                        and time.monotonic() - t_start > start_timeout):
                    return None
        finally:
            stream.stop()
            stream.close()

        if not buffer:
            return None

        audio = np.concatenate(buffer).astype(np.float32)
        duration = len(audio) / self.sample_rate
        if duration < self.min_speech_duration:
            return None

        return self.transcribe(audio)