import time
import threading
import numpy as np

class LipSyncState:
    def __init__(self, windown_ms: int = 30):
        self.windown_ms = windown_ms
        self.envelope: np.ndarray | None = None
        self.start_time = 0.0
        self.playing = False
        self.lock = threading.Lock()

    def start(self, audio: np.ndarray, sample_rate: int) -> None:
        windown = max(1, int(sample_rate * self.windown_ms / 1000))
        n_windown = max(1, int(len(audio) / windown))
        trimmed = audio[:n_windown * windown]
        chunks = trimmed.reshape(n_windown, windown)
        envelope = np.sqrt(np.mean(chunks ** 2, axis=1))

        peak = envelope.max()
        if peak > 1e-6:
            envelope /= peak

        with self.lock:
            self.envelope = envelope
            self.start_time = time.time()
            self.playing = True

    def stop(self) -> None:
        with self.lock:
            self.playing = False

    def get_current_rms(self) -> float:
        with self.lock:
            if not self.playing or self.envelope is None:
                return 0.0
            idx = int((time.time() - self.start_time) * 1000 / self.windown_ms)
            if idx >= len(self.envelope):
                self.playing = False
                return 0.0
            return float(self.envelope[idx])




