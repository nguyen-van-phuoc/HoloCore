import chess
from pathlib import Path
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont

# CONSTANTS & BOARD CONFIGURATION
BOARD_SIZE = 720
COORD_SIZE = 32
WINDOW_SIZE = BOARD_SIZE + COORD_SIZE * 2

# BOARD COLOR SCHEME
LIGHT_SQUARE = QColor("#F0D9B5")
DARK_SQUARE = QColor("#B58863")
BACKGROUND = QColor("#202020")
SELECT_COLOR = QColor(255, 215, 0, 120)  # Gold highlight with transparency
MOVE_COLOR = QColor(50, 200, 80, 100)  # Green overlay for valid moves
CAPTURE_COLOR = QColor(220, 70, 70, 100)  # Red overlay for legal captures

# ASSETS PATH
PIECE_DIR = Path(__file__).resolve().parent.parent / "chess" / "assets" / "pieces" / "cburnett"


class ChessBoard(QWidget):
    """
    PyQt6 Custom Widget for displaying and interacting with a Chessboard.
    Handles board rendering, piece graphics, square highlights, and mouse inputs.
    """

    # Custom signal emitted when a user successfully completes a move (from_square, to_square)
    move_made = pyqtSignal(str, str)

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)
        self.selected_square = None
        self.legal_moves = []
        self.input_enabled = True
        self.renderers = {}
        self.user_uci = None
        self.load_pieces()

    def load_pieces(self):
        """Pre-loads piece SVG assets into memory using QSvgRenderer."""
        pieces = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]
        if not PIECE_DIR.exists():
            print(f"[ERROR] Piece assets directory not found: {PIECE_DIR}")
            return

        for name in pieces:
            path = PIECE_DIR / f"{name}.svg"
            renderer = QSvgRenderer(str(path))
            if renderer.isValid():
                self.renderers[name] = renderer
            else:
                print(f"[ERROR] Invalid or corrupt SVG asset: {path}")

    def mousePressEvent(self, event):
        """Handles mouse click events for selecting pieces and executing moves."""
        if not self.input_enabled or event.button() != Qt.MouseButton.LeftButton:
            return

        x, y = event.position().x(), event.position().y()
        board_x, board_y = COORD_SIZE, COORD_SIZE

        # Check if the click occurred within the playable board bounds
        if not (board_x <= x < board_x + BOARD_SIZE and board_y <= y < board_y + BOARD_SIZE):
            return

        square_size = BOARD_SIZE / 8
        col = int((x - board_x) / square_size)
        row = int((y - board_y) / square_size)
        square = self.board_to_square(row, col)

        # First click: Select a piece belonging to the active player
        if self.selected_square is None:
            piece = self.game.get_piece_at(square)
            if piece and piece.color == self.game.board.turn:
                self.select_square(square)
            return

        from_square = self.selected_square

        # Deselect if clicking the same square twice
        if square == from_square:
            self.clear_selection()
            return

        # Switch selection if clicking another friendly piece
        clicked_piece = self.game.get_piece_at(square)
        if clicked_piece and clicked_piece.color == self.game.board.turn:
            self.select_square(square)
            return

        # Ignore click if it's not a legal destination
        if square not in self.legal_moves:
            self.clear_selection()
            return

        # Attempt move execution
        if self.game.make_move(from_square, square):
            self.clear_selection()
            self.move_made.emit(from_square, square)
        else:
            self.clear_selection()

    def select_square(self, square):
        """Highlights the selected square and calculates legal destination squares."""
        self.selected_square = square
        self.legal_moves = self.game.get_legal_moves(square)
        self.update()

    def clear_selection(self):
        """Resets active selections and legal move indicators."""
        self.selected_square = None
        self.legal_moves = []
        self.update()

    @staticmethod
    def board_to_square(row: int, col: int) -> str:
        """Converts array grid coordinates (row, col) to algebraic notation (e.g., 'e4')."""
        return "abcdefgh"[col] + str(8 - row)

    @staticmethod
    def square_to_board(square: str) -> tuple[int, int]:
        """Converts algebraic notation (e.g., 'e4') to array grid coordinates (row, col)."""
        return 8 - int(square[1]), ord(square[0]) - ord("a")

    @staticmethod
    def piece_to_asset(piece: chess.Piece) -> str:
        """Maps a chess.Piece object to its corresponding SVG file prefix (e.g., 'wK')."""
        color = "w" if piece.color else "b"
        pieces = {
            chess.KING: "K",
            chess.QUEEN: "Q",
            chess.ROOK: "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
            chess.PAWN: "P"
        }
        return color + pieces[piece.piece_type]

    def paintEvent(self, event):
        """Main rendering pipeline called by Qt's update system."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background window color
        painter.fillRect(self.rect(), BACKGROUND)

        # Draw UI layers in sequential order
        self.draw_board(painter, COORD_SIZE, COORD_SIZE)
        self.draw_selection(painter, COORD_SIZE, COORD_SIZE)
        self.draw_coordinates(painter, COORD_SIZE, COORD_SIZE)
        self.draw_pieces(painter, COORD_SIZE, COORD_SIZE)
        painter.end()

    def draw_board(self, painter: QPainter, board_x: float, board_y: float):
        """Draws the alternating 8x8 checkerboard squares."""
        sq = BOARD_SIZE / 8
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                painter.fillRect(QRectF(board_x + col * sq, board_y + row * sq, sq, sq), color)

    def draw_selection(self, painter: QPainter, board_x: float, board_y: float):
        """Draws square highlights for piece selections and move targets."""
        sq = BOARD_SIZE / 8

        # Highlight currently selected square
        if self.selected_square:
            row, col = self.square_to_board(self.selected_square)
            painter.fillRect(QRectF(board_x + col * sq, board_y + row * sq, sq, sq), SELECT_COLOR)

        # Highlight legal target squares
        for square in self.legal_moves:
            row, col = self.square_to_board(square)
            color = CAPTURE_COLOR if self.game.get_piece_at(square) else MOVE_COLOR
            painter.fillRect(QRectF(board_x + col * sq, board_y + row * sq, sq, sq), color)

    def draw_coordinates(self, painter: QPainter, board_x: float, board_y: float):
        """Draws rank numbers (1-8) and file letters (a-h) around the board border."""
        sq = BOARD_SIZE / 8
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        # Render file labels (a-h) on top and bottom borders
        for col, file in enumerate("abcdefgh"):
            color = DARK_SQUARE if col % 2 == 0 else LIGHT_SQUARE
            painter.setPen(color)
            painter.drawText(
                QRectF(board_x + col * sq, board_y + BOARD_SIZE, sq, COORD_SIZE),
                Qt.AlignmentFlag.AlignCenter, file
            )
            painter.drawText(
                QRectF(board_x + col * sq, 0, sq, COORD_SIZE),
                Qt.AlignmentFlag.AlignCenter, file
            )

        # Render rank labels (1-8) on left and right borders
        for row, rank in enumerate("87654321"):
            color = DARK_SQUARE if row % 2 == 0 else LIGHT_SQUARE
            painter.setPen(color)
            painter.drawText(
                QRectF(0, board_y + row * sq, COORD_SIZE, sq),
                Qt.AlignmentFlag.AlignCenter, rank
            )
            painter.drawText(
                QRectF(board_x + BOARD_SIZE, board_y + row * sq, COORD_SIZE, sq),
                Qt.AlignmentFlag.AlignCenter, rank
            )

    def draw_pieces(self, painter: QPainter, board_x: float, board_y: float):
        """Renders SVG piece graphics on top of active board squares."""
        sq = BOARD_SIZE / 8
        padding = 2
        for row in range(8):
            for col in range(8):
                piece = self.game.get_piece_at(self.board_to_square(row, col))
                if piece:
                    renderer = self.renderers.get(self.piece_to_asset(piece))
                    if renderer:
                        renderer.render(
                            painter,
                            QRectF(
                                board_x + col * sq + padding,
                                board_y + row * sq + padding,
                                sq - padding * 2,
                                sq - padding * 2
                            )
                        )