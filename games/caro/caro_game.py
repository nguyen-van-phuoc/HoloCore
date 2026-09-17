import re
from typing import Optional, Tuple

from PyQt6.QtCore import QRect, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget, QApplication

from core.app_context import AppContext
from games.caro.controller import CaroController

# =============================================================================
# 1. UI CONFIGURATION & CONSTANTS
# =============================================================================

# Board & Window Dimensions
BOARD_SIZE = 50  # Number of cells per row/column
CELL_SIZE = 18  # Pixel size of each grid cell
PANEL_WIDTH = 320  # Width of the side panel in pixels
BOARD_PX = BOARD_SIZE * CELL_SIZE
WINDOW_WIDTH = PANEL_WIDTH + BOARD_PX
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE

# Color Palette (RGB / RGBA tuples matching the original Pygame design)
COLOR_BG = (250, 251, 253)
COLOR_PANEL = (22, 27, 41)
COLOR_PANEL_HEADER = (31, 37, 56)
COLOR_PANEL_BORDER = (48, 56, 80)

COLOR_GRID = (188, 194, 206)
COLOR_GRID_BOLD = (140, 147, 166)  # Accent grid lines every 5 cells

COLOR_HOVER = (150, 190, 245, 90)  # Translucent cell highlight (RGBA)
COLOR_LAST_MOVE = (255, 190, 40)  # Highlight border for recent moves

COLOR_X = (224, 54, 68)  # Player mark color (Red/X)
COLOR_O = (33, 128, 226)  # AI mark color (Blue/O)

COLOR_TEXT_LIGHT = (232, 235, 241)
COLOR_TEXT_DIM = (150, 158, 176)
COLOR_ACCENT = (0, 214, 143)
COLOR_WIN_CARD = (255, 255, 255)
COLOR_WIN_SHADE = (10, 12, 20, 130)  # Semi-transparent overlay shade

# Prioritized list of system fonts with strong Vietnamese diacritics support
VIETNAMESE_FONTS = [
    "segoe ui", "arial", "tahoma", "dejavu sans",
    "liberation sans", "noto sans", "freesans",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vn_font(size: int, bold: bool = False) -> QFont:
    """Finds a available system font supporting Vietnamese unicode characters.

    Uses pixel sizes instead of point sizes to ensure text dimensions match
    cell measurements accurately across different platforms.

    Args:
        size: Font size in pixels.
        bold: Enables bold weight if True.

    Returns:
        A QFont instance matching the best available family.
    """
    families = QFontDatabase.families()
    for wanted in VIETNAMESE_FONTS:
        for family in families:
            if wanted in family.lower():
                font = QFont(family)
                font.setPixelSize(size)
                font.setBold(bold)
                return font

    print("[Warning] No preferred Vietnamese font found on system. Text accents may fail.")
    font = QFont()
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def current_player(game: CaroController) -> int:
    """Determines the current player's turn based on piece counts (Player X goes first).

    Args:
        game: Active CaroController instance.

    Returns:
        1 for Player (X), 2 for AI (O).
    """
    x_count = sum(row.count(1) for row in game.board)
    o_count = sum(row.count(2) for row in game.board)
    return 1 if x_count == o_count else 2


def wrap_text(text: str, font: QFont, max_width: int) -> list:
    """Wraps text into multiple lines fitting within a maximum pixel width.

    Args:
        text: Input string to wrap.
        font: QFont used to measure character dimensions.
        max_width: Maximum allowed width in pixels.

    Returns:
        List of strings, each fitting within max_width.
    """
    metrics = QFontMetrics(font)
    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        if metrics.horizontalAdvance(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# =============================================================================
# 2. WORKER THREAD FOR AI MOVES
# =============================================================================

class AIWorker(QThread):
    """Background worker thread executing AI move computations.

    Prevents Ollama API requests/network calls from freezing the main GUI thread.
    """
    moveReady = pyqtSignal(int, int, str)  # Emits (row, col, ai_dialogue)
    failed = pyqtSignal(str)  # Emits error message string

    def __init__(self, game: CaroController, parent=None) -> None:
        super().__init__(parent)
        self.game = game

    def run(self) -> None:
        """Executes the LLM move computation in the background thread."""
        try:
            row, col, dialogue = self.game.get_llm_move()
            self.moveReady.emit(row, col, dialogue)
        except Exception as exc:
            self.failed.emit(str(exc))


# =============================================================================
# 3. GAME BOARD WIDGET
# =============================================================================

class BoardWidget(QWidget):
    """Custom widget responsible for rendering the grid, mouse interactions, and marks."""

    cellClicked = pyqtSignal(int, int)  # Emits clicked board coordinates (row, col)

    def __init__(self, game: CaroController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.game = game
        self.enabled_for_input = True
        self.hover_cell: Optional[Tuple[int, int]] = None

        self.setFixedSize(BOARD_PX, BOARD_PX)
        self.setMouseTracking(True)

        self.font_symbol = get_vn_font(CELL_SIZE - 4, bold=True)
        # Pre-render static background grid to optimize repaint performance
        self.grid_pixmap = self._build_grid_pixmap()

    def _build_grid_pixmap(self) -> QPixmap:
        """Draws static grid lines onto a cached QPixmap.

        Avoids re-drawing over 200 grid lines on every paint event cycle.
        """
        pixmap = QPixmap(BOARD_PX, BOARD_PX)
        pixmap.fill(QColor(*COLOR_BG))
        painter = QPainter(pixmap)

        for i in range(BOARD_SIZE + 1):
            bold = (i % 5 == 0)
            color = QColor(*(COLOR_GRID_BOLD if bold else COLOR_GRID))
            painter.setPen(QPen(color, 2 if bold else 1))
            # Horizontal and vertical lines
            painter.drawLine(0, i * CELL_SIZE, BOARD_PX, i * CELL_SIZE)
            painter.drawLine(i * CELL_SIZE, 0, i * CELL_SIZE, BOARD_PX)

        painter.end()
        return pixmap

    def _cell_at(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Translates pixel coordinates to board grid indices (row, col)."""
        col, row = x // CELL_SIZE, y // CELL_SIZE
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None

    def paintEvent(self, event) -> None:
        """Main rendering pipeline for the board, hover effects, and game pieces."""
        painter = QPainter(self)

        # 1. Draw cached background grid
        painter.drawPixmap(0, 0, self.grid_pixmap)

        # 2. Render hover highlight if active
        if self.enabled_for_input and self.game.winner == 0 and self.hover_cell:
            r, c = self.hover_cell
            hover_color = QColor(*COLOR_HOVER[:3])
            hover_color.setAlpha(COLOR_HOVER[3])
            painter.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE, hover_color)

        # 3. Highlight recent moves with rounded rectangular borders
        painter.setPen(QPen(QColor(*COLOR_LAST_MOVE), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for move in (self.game.last_move_player, self.game.last_move_ai):
            if move:
                r, c = move
                painter.drawRoundedRect(
                    c * CELL_SIZE + 1, r * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2, 3, 3
                )

        # 4. Render X and O marks on the board
        painter.setFont(self.font_symbol)
        for r in range(BOARD_SIZE):
            row = self.game.board[r]
            for c in range(BOARD_SIZE):
                val = row[c]
                if val:
                    painter.setPen(QColor(*COLOR_X) if val == 1 else QColor(*COLOR_O))
                    rect = QRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "X" if val == 1 else "O")

        painter.end()

    def mouseMoveEvent(self, event) -> None:
        """Tracks cursor position to update cell hover states."""
        pos = event.position().toPoint()
        new_hover = self._cell_at(pos.x(), pos.y())
        if new_hover != self.hover_cell:
            self.hover_cell = new_hover
            self.update()

    def leaveEvent(self, event) -> None:
        """Clears cursor highlight when leaving the board area."""
        if self.hover_cell is not None:
            self.hover_cell = None
            self.update()

    def mousePressEvent(self, event) -> None:
        """Emits signal on valid board clicks."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.enabled_for_input or self.game.winner != 0:
            return
        pos = event.position().toPoint()
        cell = self._cell_at(pos.x(), pos.y())
        if cell:
            self.cellClicked.emit(*cell)


# =============================================================================
# 4. SIDE PANEL WIDGET
# =============================================================================

class PanelWidget(QWidget):
    """Side panel displaying game title, AI dialogue bubble, turn status, and key controls."""

    def __init__(self, game: CaroController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.game = game
        self.dialogue = ""
        self.setFixedSize(PANEL_WIDTH, WINDOW_HEIGHT)

        # Pre-initialize typography styles
        self.font_title = get_vn_font(22, bold=True)
        self.font_chat = get_vn_font(16)
        self.font_label = get_vn_font(13, bold=True)
        self.font_small = get_vn_font(13)

    def set_dialogue(self, text: str) -> None:
        """Updates AI text box dialogue and triggers a redraw."""
        self.dialogue = text
        self.update()

    def paintEvent(self, event) -> None:
        """Renders side panel UI elements."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # --- Base background & Header ---
        painter.fillRect(self.rect(), QColor(*COLOR_PANEL))
        painter.fillRect(0, 0, PANEL_WIDTH, 80, QColor(*COLOR_PANEL_HEADER))
        painter.setPen(QPen(QColor(*COLOR_PANEL_BORDER), 2))
        painter.drawLine(PANEL_WIDTH - 1, 0, PANEL_WIDTH - 1, WINDOW_HEIGHT)

        # Header Title
        painter.setFont(self.font_title)
        painter.setPen(QColor(*COLOR_ACCENT))
        painter.drawText(QRect(20, 15, PANEL_WIDTH - 40, 50),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Caro With Holo")
        painter.setPen(QPen(QColor(*COLOR_ACCENT), 2))
        painter.drawLine(20, 78, PANEL_WIDTH - 20, 78)

        # --- AI Speech Dialogue Box ---
        box = QRect(18, 112, PANEL_WIDTH - 36, 220)
        painter.setPen(QPen(QColor(*COLOR_ACCENT), 1))
        painter.setBrush(QColor(*COLOR_PANEL_HEADER))
        painter.drawRoundedRect(box, 10, 10)

        # Dialogue Header Tag
        painter.setFont(self.font_label)
        tag_text = "Holo nói"
        tag_w = QFontMetrics(self.font_label).horizontalAdvance(tag_text) + 16
        tag_rect = QRect(box.x() + 14, box.y() - 11, tag_w, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*COLOR_ACCENT))
        painter.drawRoundedRect(tag_rect, 11, 11)
        painter.setPen(QColor(*COLOR_PANEL))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag_text)

        # Render Dialogue Text
        chat_rect = QRect(box.x() + 16, box.y() + 18, box.width() - 32, box.height() - 34)
        painter.setFont(self.font_chat)
        painter.setPen(QColor(*COLOR_TEXT_LIGHT))
        metrics = QFontMetrics(self.font_chat)
        lines = wrap_text(f"{self.dialogue}", self.font_chat, chat_rect.width())
        line_height = metrics.lineSpacing() + 4
        max_lines = max(1, chat_rect.height() // line_height)
        y = chat_rect.y() + metrics.ascent()

        for line in lines[:max_lines]:
            painter.drawText(chat_rect.x(), y, line)
            y += line_height

        # --- Player Turn Cards ---
        turn = current_player(self.game) if self.game.winner == 0 else 0
        badge_y = box.bottom() + 18
        badge_w = (PANEL_WIDTH - 36 - 12) // 2
        self._draw_turn_badge(painter, QRect(18, badge_y, badge_w, 44), "BẠN (X)", QColor(*COLOR_X), turn == 1)
        self._draw_turn_badge(painter, QRect(18 + badge_w + 12, badge_y, badge_w, 44),
                              "AI (O)", QColor(*COLOR_O), turn == 2)

        # Total Moves Counter
        moves = sum(row.count(1) + row.count(2) for row in self.game.board)
        painter.setFont(self.font_small)
        painter.setPen(QColor(*COLOR_TEXT_DIM))
        painter.drawText(QRect(18, badge_y + 58, PANEL_WIDTH - 36, 20),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"Số nước đã đánh: {moves}")

        # --- Color Legend Section ---
        self._draw_legend(painter, badge_y + 92)

        # Game reset instructions footer
        painter.setFont(self.font_small)
        painter.setPen(QColor(*COLOR_TEXT_DIM))
        painter.drawText(QRect(18, WINDOW_HEIGHT - 34, PANEL_WIDTH - 36, 20),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         "Nhấn phím R để chơi lại khi kết thúc ván")

        painter.end()

    def _draw_turn_badge(self, painter: QPainter, rect: QRect, label: str, color: QColor, active: bool) -> None:
        """Renders status badges indicating current active player."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*COLOR_PANEL_HEADER))
        painter.drawRoundedRect(rect, 8, 8)

        border_color = color if active else QColor(*COLOR_PANEL_BORDER)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border_color, 2 if active else 1))
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QColor(*COLOR_TEXT_LIGHT) if active else QColor(*COLOR_TEXT_DIM))
        painter.setFont(self.font_label)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_legend(self, painter: QPainter, y: int) -> None:
        """Renders color indicators legend."""
        x = 18
        painter.setFont(self.font_label)
        painter.setPen(QColor(*COLOR_TEXT_DIM))
        painter.drawText(QRect(x, y, PANEL_WIDTH - 36, 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "CHÚ THÍCH")
        y += 24

        items = (
            (QColor(*COLOR_X), "Nước đi của bạn (X)"),
            (QColor(*COLOR_O), "Nước đi của AI (O)"),
            (QColor(*COLOR_LAST_MOVE), "Viền vàng: nước đi gần nhất"),
        )
        painter.setFont(self.font_small)
        for color, label in items:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(x, y, 14, 14, 3, 3)
            painter.setPen(QColor(*COLOR_TEXT_LIGHT))
            painter.drawText(QRect(x + 22, y - 3, PANEL_WIDTH - 36 - 22, 20),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            y += 24


# =============================================================================
# 5. WIN/GAME OVER OVERLAY
# =============================================================================

class WinOverlay(QWidget):
    """Transparent overlay displaying match outcome modal over the game board."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.message = ""
        self.color = QColor(*COLOR_TEXT_DIM)
        self.font_title = get_vn_font(44, bold=True)
        self.font_sub = get_vn_font(17)

        # Allow user click events to pass through background elements if needed
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_result(self, winner: int) -> None:
        """Sets result banner styling based on winner ID (1: Player, 2: AI, Else: Draw)."""
        if winner == 1:
            self.message, self.color = "BẠN THẮNG!", QColor(*COLOR_X)
        elif winner == 2:
            self.message, self.color = "HOLO THẮNG!", QColor(*COLOR_O)
        else:
            self.message, self.color = "HÒA CỜ!", QColor(*COLOR_TEXT_DIM)
        self.update()

    def paintEvent(self, event) -> None:
        """Renders semi-transparent screen overlay and victory card."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background overlay
        shade = QColor(*COLOR_WIN_SHADE[:3])
        shade.setAlpha(COLOR_WIN_SHADE[3])
        painter.fillRect(self.rect(), shade)

        # Centered result card modal
        card = QRect(0, 0, 380, 170)
        card.moveCenter(self.rect().center())

        painter.setPen(QPen(self.color, 3))
        painter.setBrush(QColor(*COLOR_WIN_CARD))
        painter.drawRoundedRect(card, 16, 16)

        # Main result message text
        painter.setFont(self.font_title)
        painter.setPen(self.color)
        painter.drawText(QRect(card.x(), card.center().y() - 50, card.width(), 60),
                         Qt.AlignmentFlag.AlignCenter, self.message)

        # Restart hint subtitle
        painter.setFont(self.font_sub)
        painter.setPen(QColor(95, 100, 110))
        painter.drawText(QRect(card.x(), card.center().y() + 10, card.width(), 40),
                         Qt.AlignmentFlag.AlignCenter, "Nhấn phím R để chơi lại")

        painter.end()


# =============================================================================
# 6. MAIN APPLICATION WINDOW
# =============================================================================

class CaroWindow(QWidget):
    """Main window orchestrating child widgets, worker threads, and game loops."""

    game_closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.tts = AppContext.tts

        self.setWindowTitle("Caro 50x50 - With Holo")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.game = CaroController(board_size=BOARD_SIZE)
        self.is_ai_turn = False
        self.ai_worker: Optional[AIWorker] = None

        # Instantiate side panel
        self.panel = PanelWidget(self.game, self)
        self.panel.move(0, 0)

        # Instantiate board widget
        self.board_widget = BoardWidget(self.game, self)
        self.board_widget.move(PANEL_WIDTH, 0)
        self.board_widget.cellClicked.connect(self.on_cell_clicked)

        # Instantiate game-over overlay
        self.overlay = WinOverlay(self)
        self.overlay.setGeometry(PANEL_WIDTH, 0, BOARD_PX, BOARD_PX)
        self.overlay.hide()

        self.panel.set_dialogue("Xin chào! Bạn đánh quân X trước nhé.")
        self.setFocus()

    def on_cell_clicked(self, row: int, col: int) -> None:
        """Handles user board selection events."""
        if self.is_ai_turn or self.game.winner != 0:
            return

        if self.game.play_move(row, col, player=1):
            self.board_widget.update()
            self.panel.update()
            if self.game.winner == 0:
                self.start_ai_turn()
            else:
                self.show_result()

    def start_ai_turn(self) -> None:
        """Disables board input and spawns AI thread computation."""
        self.is_ai_turn = True
        self.board_widget.enabled_for_input = False
        self.panel.set_dialogue("Đang suy tính nước đi...")

        # Spawn background execution thread to query Ollama model asynchronously
        self.ai_worker = AIWorker(self.game, self)
        self.ai_worker.moveReady.connect(self.on_ai_move_ready)
        self.ai_worker.failed.connect(self.on_ai_move_failed)
        self.ai_worker.start()

    def on_ai_move_ready(self, row: int, col: int, dialogue: str) -> None:
        """Callback executed when AI computation succeeds.

        Note: Board state updates occur strictly within the main thread.
        """
        self.game.play_move(row, col, player=2)
        self.is_ai_turn = False
        self.board_widget.enabled_for_input = True

        # Trigger TTS audio and strip animation markup tags for text display
        if dialogue:
            self.tts.speak(dialogue)

        self.panel.set_dialogue(re.sub(r"<motion:[^>]+>", "", dialogue))

        print(dialogue)
        self.board_widget.update()
        self.panel.update()

        if self.game.winner != 0:
            self.show_result()

    def on_ai_move_failed(self, error_message: str) -> None:
        """Callback handling AI worker errors (e.g., connection lost to Ollama)."""
        self.is_ai_turn = False
        self.board_widget.enabled_for_input = True
        self.panel.set_dialogue(f"AI gặp lỗi khi kết nối Ollama: {error_message}")

    def show_result(self) -> None:
        """Displays game-over modal overlay."""
        self.overlay.set_result(self.game.winner)
        self.overlay.show()
        self.overlay.raise_()

    def restart_game(self) -> None:
        """Resets controller state and clears game interface."""
        self.game = CaroController(board_size=BOARD_SIZE)
        self.is_ai_turn = False
        self.board_widget.game = self.game
        self.board_widget.enabled_for_input = True
        self.board_widget.hover_cell = None
        self.panel.game = self.game
        self.overlay.hide()
        self.panel.set_dialogue("Ván cờ mới! Bạn đi trước nhé.")
        self.board_widget.update()
        self.panel.update()

    def keyPressEvent(self, event) -> None:
        """Handles global key inputs (R to restart game upon game over)."""
        if event.key() == Qt.Key.Key_R and self.game.winner != 0:
            self.restart_game()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Ensures worker thread is safely terminated prior to window destruction."""
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self.ai_worker.quit()
            self.ai_worker.wait(2000)
        self.game_closed.emit()
        super().closeEvent(event)