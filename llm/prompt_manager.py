from pathlib import Path

class PromptManager:
    def __init__(self):
        self.path_system = str(Path(__file__).resolve().parent / "system_prompt.txt")
        self.path_tools = str(Path(__file__).resolve().parent.parent / "tools" / "tools_prompt.txt")

    def load_system(self):
        with open(self.path_system, "r", encoding="utf-8") as f:
            return f.read()

    def load_tools(self):
        with open(self.path_tools, "r", encoding="utf-8") as f:
            return f.read()

    def get_system_prompt(self):
        return self.load_system()

    def get_tools_prompt(self):
        return self.load_tools()

    def reload(self):
        self.load_system()
        self.load_tools()
