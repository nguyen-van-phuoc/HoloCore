class MemoryShort:
    def __init__(self):
        self.MEMORY_SHORT = []
        self.MAX_MEMORY_SHORT = 20

    def get_memory_short(self):
        return self.MEMORY_SHORT[-self.MAX_MEMORY_SHORT:]

    def add_momory_short(self, text):
        self.MEMORY_SHORT.append(text)