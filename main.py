from pipeline import PipeLine
from core.app_context import AppContext

class Main:
    def __init__(self):
        AppContext.init()
        self.pipline = PipeLine()

if __name__ == "__main__":
    app = Main()
