import re

class Chunker:
    def __init__(self):
        self._buf = ""
        self.max_chunk_len = 256

    def feed(self, token: str):
        self._buf += token

        if re.search(r'[。.,！!？?—\n]|(?<!\.)\.(?!\.)', self._buf):
            return self._flush()

        if len(self._buf) >= self.max_chunk_len:
            return self._flush()
        return None

    def flush_remaining(self):
        if self._buf.strip():
            return self._flush()
        return None

    def _flush(self):
        chunk = self._buf.strip()
        self._buf = ""
        if len(chunk) > 1 or chunk.isalnum():
            return chunk
        return None