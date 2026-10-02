from llm.chunker import Chunker
from tts.manager_tts import TTSManager

class StreamProcessor:
    def __init__(self, tts: TTSManager) -> None:
        self.tts = tts
        self.full_response = ""
        self._chunker = Chunker()

    def process_token(self, token: str) -> None:
        if not token:
            return
        self.full_response += token
        print(token, end="", flush=True)
        chunk = self._chunker.feed(token)
        if chunk:
            self.tts.speak(chunk)

    def flush(self) -> None:
        remaining = self._chunker.flush_remaining()
        if remaining:
            self.tts.speak(remaining)
