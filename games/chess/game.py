import re
import games.chess.chess_config as config
from games.chess.chess_game import ChessGame
from games.chess.flow_layout import FlowLayout
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from games.chess.chess_board import ChessBoard, WINDOW_SIZE
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QApplication, QGridLayout, QLayout
from games.chess.chess_ai import OllamaClient, CharacterAgent, ChessGameController

from tts.gpt_sovits import GPTSoVits


# AI WORKER (BACKGROUND THREAD TO PREVENT UI FREEZING)
class AIWorker(QThread):
    """Worker thread running AI evaluation asynchronously to keep UI responsive."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ai_controller):
        super().__init__()
        self.ai_controller = ai_controller

    def run(self):
        try:
            # AI evaluation occurs off the main thread
            result = self.ai_controller.play_agent_turn()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QWidget):
    """Main application window for playing chess against the AI persona."""

    game_losed = pyqtSignal()

    def __init__(self, tts_engine: GPTSoVits):
        super().__init__()
        self.setWindowTitle("Play Chess with Holo")

        self.game = ChessGame()
        self.client = OllamaClient(
            base_url=config.OLLAMA_URL,
            model_name=config.OLLAMA_MODEL,
            max_retries=config.LLM_MAX_RETRIES,
            timeout=config.LLM_TIMEOUT
        )
        self.agent = CharacterAgent(llm_client=self.client, system_prompt=config.SYSTEM_PROMPT)

        self.ai_controller = ChessGameController(
            board=self.game.board,
            agent=self.agent,
            stockfish_path=config.STOCKFISH_PATH,
            engine_time_limit=config.STOCKFISH_TIME,
            stockfish_threads=config.STOCKFISH_THREADS,
            stockfish_hash=config.STOCKFISH_HASH
        )

        try:
            self.ai_controller.start()
            print("Stockfish engine started successfully.")
        except Exception as e:
            print(f"Could not start Stockfish engine: {e}")

        self.tts = tts_engine

        self.setup_ui()
        self.update_status()

    def closeEvent(self, event):
        """Cleanly shuts down background threads and engine sessions on window close."""
        print("[GAME] closeEvent called.")
        self.game_losed.emit()
        self.ai_controller.stop()
        self.client.close()
        event.accept()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #202020; color: #eeeeee; }
            QPushButton { background-color: #333333; border: 1px solid #555555; border-radius: 6px; padding: 8px 16px; }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
        """)

        self.board_widget = ChessBoard(self.game)
        self.status_label = QLabel("White to move")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.eval_qwidget = QWidget()
        self.eval_qwidget.setObjectName("evalContainer")

        self.eval_move_layout = FlowLayout(self.eval_qwidget, spacing=5)
        self.eval_move_layout.setContentsMargins(10, 10, 10, 10)

        self.talk_label = QLabel()
        self.talk_label.setWordWrap(True)
        self.talk_label.setMinimumWidth(300)

        self.tactic_label = QLabel("")
        self.tactic_label.setWordWrap(True)
        self.tactic_label.setMinimumWidth(300)
        self.tactic_label.setStyleSheet("""
            QLabel {
                color: #F5F5F5;
                background-color: rgba(45, 35, 35, 220);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 14px;
                padding: 10px 16px;
                font-size: 15px;
                font-weight: 500;
            }
        """)

        self.reset_button = QPushButton("New Game")
        self.undo_button = QPushButton("Undo")

        self.history_widget = QWidget()
        self.move_history_layout = FlowLayout(self.history_widget, spacing=5)

        self.label_history = QLabel("HIS")
        self.label_history.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 160);
                color: #222222;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 7px;
                padding: 5px 12px;
            }
        """)
        self.move_history_layout.addWidget(self.label_history)

        # Layout hierarchy
        board_layout = QVBoxLayout()
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(self.board_widget)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(self.eval_qwidget)
        side_layout.addWidget(self.talk_label)
        side_layout.addWidget(self.tactic_label)
        side_layout.addStretch()

        side_layout.addWidget(self.history_widget)
        side_layout.addWidget(self.undo_button)
        side_layout.addWidget(self.reset_button)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.addLayout(board_layout)
        main_layout.addLayout(side_layout)

        # Event connections
        self.board_widget.move_made.connect(self.on_player_move)
        self.reset_button.clicked.connect(self.reset_game)
        self.undo_button.clicked.connect(self.undo_move)

        self.resize(WINDOW_SIZE + 350, WINDOW_SIZE + 30)

    def on_player_move(self, from_square, to_square):
        self.update_status()
        self.add_move_history(from_square + to_square, llm=False)
        if self.game.is_game_over():
            self.handle_game_over()
            return

        if self.game.is_black_turn():
            self.board_widget.input_enabled = False
            self.status_label.setText("Holo is thinking...")
            # Trigger AI move worker after 100ms delay
            QTimer.singleShot(100, self.ai_move)

    def ai_move(self):
        """Spawns background worker thread to process AI turn."""
        print("AI started processing on background thread...")
        self.ai_thread = AIWorker(self.ai_controller)
        self.ai_thread.finished.connect(self.on_ai_done)
        self.ai_thread.error.connect(self.on_ai_error)
        self.ai_thread.start()

    def on_ai_done(self, result):
        """Callback triggered when AI finishes move computation."""
        print(f"AI MOVE: {result.move_uci}")
        print(f"AI TACTIC: {result.tactic}")
        print(f"AI TALK: {result.talk}")

        self.add_move_history(result.move_uci, llm=True)

        if not result.talk:
            self.talk_label.setStyleSheet("""
                QLabel {
                    color: #F5F5F5;
                    background: transparent;
                    border: none;
                    padding: 0px;
                    margin: 0px;
                    font-size: 15px;
                    font-weight: 500;
                }
            """)
            self.talk_label.setText("")
        else:
            self.talk_label.setStyleSheet("""
                QLabel {
                    color: #F5F5F5;
                    background-color: rgba(35, 35, 45, 220);
                    border: 1px solid rgba(255, 255, 255, 40);
                    border-radius: 14px;
                    padding: 10px 16px;
                    font-size: 15px;
                    font-weight: 500;
                }
            """)
            clean_text = re.sub(r"<motion:[^>]+>", "", result.talk)
            self.talk_label.setText(f"Holo: {clean_text}")

        self.eval_qwidget.setStyleSheet("""
            QWidget#evalContainer {
                background-color: rgba(20, 20, 20, 150);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 10px;
            }
        """)

        self.add_move_evaluation(result.eval, self.eval_move_layout)

        # Update dialogue & tactical analysis text
        self.tactic_label.setText(f"Phân tích chiến thuật: {result.tactic}")

        # Refresh board UI
        self.board_widget.clear_selection()
        self.board_widget.update()
        self.update_status()

        if self.game.is_game_over():
            self.handle_game_over()
        else:
            self.board_widget.input_enabled = True

        self.tts.speak(result.talk)

    def on_ai_error(self, error_msg):
        """Callback triggered if AI thread encounters an exception."""
        print(f"AI ERROR: {error_msg}")
        self.tactic_label.setText(f"AI Error: {error_msg}")

        self.board_widget.input_enabled = True
        self.update_status()

    def update_status(self):
        """Updates game status label based on current board state."""
        if self.game.is_checkmate():
            self.status_label.setText("Checkmate!")
        elif self.game.is_stalemate():
            self.status_label.setText("Draw - Stalemate")
        elif self.game.is_check():
            self.status_label.setText("White is in check!" if self.game.is_white_turn() else "Black is in check!")
        else:
            self.status_label.setText("Your Turn - White" if self.game.is_white_turn() else "AI's Turn - Black")

    def handle_game_over(self):
        """Displays final game outcome."""
        if self.game.is_checkmate():
            self.status_label.setText("Checkmate - Holo Wins" if self.game.is_white_turn() else "Checkmate - You Win")
        elif self.game.is_stalemate():
            self.status_label.setText("Draw - Stalemate")
        else:
            self.status_label.setText("Game Over")

    def undo_move(self):
        """Undoes the previous full turn (both AI move and player move)."""
        if not self.game.board.move_stack:
            return

        self.game.undo()  # Undo AI move
        if self.game.board.move_stack:
            self.game.undo()  # Undo Player move

        self.board_widget.clear_selection()
        self.tactic_label.clear()
        self.board_widget.input_enabled = True
        self.update_status()

    def reset_game(self):
        """Resets board to initial setup."""
        self.game.reset()
        self.board_widget.clear_selection()
        self.board_widget.input_enabled = True
        self.tactic_label.clear()
        self.update_status()

    def add_move_history(self, uci: str, llm: bool):
        """Appends move label to visual move history flow layout."""
        label = QLabel(uci)
        if llm:
            label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 0, 0, 220);
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: 500;
                    border: 1px solid rgba(189, 0, 0, 120);
                    border-radius: 7px;
                    padding: 5px 10px;
                }
            """)
        else:
            label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 255, 55, 220);
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: 500;
                    border: 1px solid rgba(122, 255, 151, 120);
                    border-radius: 7px;
                    padding: 5px 10px;
                }
            """)
        count = self.move_history_layout.count()
        if count >= 28:
            self.move_history_layout.takeAt(1)
        self.move_history_layout.addWidget(label)

    def add_move_evaluation(self, eval_ls: list, layout: QLayout):
        """Displays top candidate move evaluations evaluated by Stockfish."""
        if layout is None:
            return

        # Clear existing evaluations
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.deleteLater()

        # "VAL" Header label
        label_vals = QLabel("VAL")
        label_vals.setStyleSheet("""
            QLabel {
                color: #222222;
                background-color: rgba(255, 255, 255, 160);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 7px;
                padding: 5px 12px;
                margin: 0px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(label_vals)

        # Render each candidate move score
        for ev in eval_ls:
            move_uci = ev.move_uci
            score_cp = int(ev.score_cp)

            if score_cp > 0:
                eval_label = QLabel(f"{move_uci}: +{score_cp} cp")
                eval_label.setStyleSheet("""
                    QLabel {
                        color: #D8FFD8;
                        background-color: rgba(40, 120, 60, 160);
                        border: 1px solid rgba(100, 220, 120, 100);
                        border-radius: 7px;
                        padding: 4px 8px;
                        margin: 0px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)
            elif score_cp < 0:
                eval_label = QLabel(f"{move_uci}: {score_cp} cp")
                eval_label.setStyleSheet("""
                    QLabel {
                        color: #FFD8D8;
                        background-color: rgba(140, 45, 45, 160);
                        border: 1px solid rgba(240, 100, 100, 100);
                        border-radius: 7px;
                        padding: 4px 8px;
                        margin: 0px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)
            else:
                eval_label = QLabel(f"{move_uci}: 0 cp")
                eval_label.setStyleSheet("""
                    QLabel {
                        color: #E8E8E8;
                        background-color: rgba(100, 100, 100, 140);
                        border: 1px solid rgba(180, 180, 180, 80);
                        border-radius: 7px;
                        padding: 4px 8px;
                        margin: 0px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)

            layout.addWidget(eval_label)