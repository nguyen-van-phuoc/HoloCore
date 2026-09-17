from core.app_context import AppContext

class GameManager:
    def __init__(self):
        self.chess_window = None
        self.caro_window = None

    def game_play(self, name):

        if name == "COMMAND:OPEN_CHESS":
            from games.chess.game import MainWindow

            if self.chess_window is None:
                self.chess_window = MainWindow(AppContext.tts)
                self.chess_window.game_losed.connect(self.on_chess_close)

            self.chess_window.show()
            self.chess_window.raise_()
            self.chess_window.activateWindow()

        elif name == "COMMAND:OPEN_CARO":
            from games.caro.caro_game import CaroWindow

            if self.caro_window is None:
                self.caro_window = CaroWindow()
                self.caro_window.game_closed.connect(self.on_caro_close)

            self.caro_window.show()
            self.caro_window.raise_()
            self.caro_window.activateWindow()

    def on_chess_close(self):
        print("[GAME] Chess closed!")
        self.chess_window = None

    def on_caro_close(self):
        print("[GAME] Caro closed!")
        self.caro_window = None

