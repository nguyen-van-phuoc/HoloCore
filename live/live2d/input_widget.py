import sys
import asyncio
from pathlib import Path

from core.app_context import AppContext
from live.live2d.strands import Strands
from PyQt6.QtWidgets import QGraphicsOpacityEffect
from live.live2d.strands import _default_surface_format
from PyQt6.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton)
from PyQt6.QtGui import (QColor, QFont, QPainter, QPainterPath, QRadialGradient, QLinearGradient, QPen, QBrush,
                         QSurfaceFormat, QIcon, QPixmap, QTransform)
from PyQt6.QtCore import (Qt, QPointF, QEvent, pyqtSignal, QPropertyAnimation, QEasingCurve, QVariantAnimation,
                          QTimer, QRectF, QRect, QObject, QThread, QSize, QCoreApplication)

from stt.manager import STTManager


class FocusLineEdit(QLineEdit):
    focusChanged = pyqtSignal(bool)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focusChanged.emit(True)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.focusChanged.emit(False)

class TrueLiquidGlassCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.setMouseTracking(True)
        self.is_focused = False
        self.is_hovered = False
        self.active_progress = 0.0

        self.target_pos = QPointF(350, 30)
        self.surface_light = QPointF(350, 30)
        self.deep_light = QPointF(350, 30)

        self.anim = QVariantAnimation(self)
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.valueChanged.connect(self._on_anim_step)

        # Chạy physics ở 60FPS
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._liquid_physics)
        self.render_timer.start(16)



    def set_focused(self, focused: bool):
        self.is_focused = focused
        self._update_state()

    def set_hovered(self, hovered: bool):
        self.is_hovered = hovered
        self._update_state()

    def _update_state(self):
        target = 1.0 if (self.is_focused or self.is_hovered) else 0.0
        self.anim.stop()
        self.anim.setStartValue(self.active_progress)
        self.anim.setEndValue(target)
        self.anim.start()

    def _on_anim_step(self, val):
        self.active_progress = val
        self.update()

    def update_light_target(self, pos: QPointF):
        self.target_pos = pos

    def mouseMoveEvent(self, event):
        self.update_light_target(event.position())
        super().mouseMoveEvent(event)

    def _liquid_physics(self):
        dx_surf = self.target_pos.x() - self.surface_light.x()
        dy_surf = self.target_pos.y() - self.surface_light.y()
        dx_deep = self.target_pos.x() - self.deep_light.x()
        dy_deep = self.target_pos.y() - self.deep_light.y()

        if abs(dx_surf) < 0.5 and abs(dy_surf) < 0.5 and abs(dx_deep) < 0.5 and abs(dy_deep) < 0.5:
            return

        self.surface_light.setX(self.surface_light.x() + dx_surf * 0.07)
        self.surface_light.setY(self.surface_light.y() + dy_surf * 0.07)
        self.deep_light.setX(self.deep_light.x() + dx_deep * 0.025)
        self.deep_light.setY(self.deep_light.y() + dy_deep * 0.025)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = rect.height() / 2.0

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)

        # 1. Transparent glass background
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        base_color = QColor(18, 20, 24, 120)
        painter.fillPath(path, QBrush(base_color))

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setClipPath(path)

        # 2. High Viscosity Grade
        deep_grad = QRadialGradient(self.deep_light, 220.0)
        deep_grad.setColorAt(0.0, QColor(255, 255, 255, 28))
        deep_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(path, QBrush(deep_grad))

        # 3. SURFACE VISCOSITY CLASS
        surface_grad = QRadialGradient(self.surface_light, 75.0)
        surface_grad.setColorAt(0.0, QColor(255, 255, 255, 75))
        surface_grad.setColorAt(0.4, QColor(255, 255, 255, 18))
        surface_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(path, QBrush(surface_grad))

        painter.setClipping(False)

        # 4. Liquid Glass Border
        p = self.active_progress
        outer_pen = QPen()
        outer_pen.setWidthF(2.0)
        outer_grad = QLinearGradient(0, rect.top(), 0, rect.bottom())

        r_top, g_top, b_top = int(255 - 10 * p), int(255 - 110 * p), int(255 - 215 * p)
        alpha_top = int(90 * (1 - p) + 255 * p)

        outer_grad.setColorAt(0.0, QColor(r_top, g_top, b_top, alpha_top))
        outer_grad.setColorAt(0.4, QColor(r_top, g_top, b_top, int(alpha_top * 0.15)))
        outer_grad.setColorAt(0.65, QColor(255, 255, 255, 8))
        outer_grad.setColorAt(1.0, QColor(255, 255, 255, 0))

        outer_pen.setBrush(QBrush(outer_grad))
        painter.setPen(outer_pen)
        painter.drawPath(path)

        inner_path = QPainterPath()
        inner_path.addRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), radius - 1, radius - 1)
        inner_grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        inner_grad.setColorAt(0.0, QColor(255, 255, 255, int(25 + 45 * p)))
        inner_grad.setColorAt(0.5, QColor(255, 255, 255, 0))

        inner_pen = QPen(QBrush(inner_grad), 1.0)
        painter.setPen(inner_pen)
        painter.drawPath(inner_path)


class WaterDropAmberButton(QPushButton):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        radius = rect.height() / 2.0
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)

        is_pressed = self.isDown()
        base_color = QColor(200, 100, 20, 200) if is_pressed else QColor(220, 130, 40, 220)
        painter.fillPath(path, QBrush(base_color))
        painter.setClipPath(path)

        glow_grad = QRadialGradient(rect.width() / 2.0, rect.height() * 0.9, radius)
        glow_grad.setColorAt(0.0, QColor(255, 220, 150, 255))
        glow_grad.setColorAt(1.0, QColor(255, 180, 80, 0))
        painter.fillPath(path, QBrush(glow_grad))

        hl_rect = QRectF(rect.x() + 5, rect.y() + 2, rect.width() - 10, rect.height() * 0.4)
        hl_path = QPainterPath()
        hl_path.addRoundedRect(hl_rect, hl_rect.height() / 2, hl_rect.height() / 2)

        hl_grad = QLinearGradient(0, hl_rect.top(), 0, hl_rect.bottom())
        hl_grad.setColorAt(0.0, QColor(255, 255, 255, 240))
        hl_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(hl_path, QBrush(hl_grad))
        painter.setClipping(False)

        painter.setPen(QPen(QColor(255, 255, 255, 120) if not is_pressed else QColor(255, 255, 255, 40), 1.0))
        painter.drawPath(path)

        painter.setPen(QColor(40, 15, 0))
        font = QFont("Segoe UI", 16, QFont.Weight.Black)
        painter.setFont(font)
        text_rect = rect.translated(0, 1) if is_pressed else rect
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())


class MicWorker(QObject):
    text_received = pyqtSignal(str)  # Phát tín hiệu mỗi khi nghe xong 1 câu
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, stt_manager):
        super().__init__()
        self.stt_manager = stt_manager
        self.is_running = True  # Cờ điều khiển vòng lặp

    def run(self):
        """Continuous listening loop until is_running = False"""
        while self.is_running:
            try:
                text = self.stt_manager.listen()
                if text and self.is_running:
                    self.text_received.emit(text.strip())
            except Exception as e:
                if self.is_running:
                    self.error.emit(str(e))
        self.finished.emit()

    def stop(self):
        """Gọi hàm này để dừng vòng lặp từ bên ngoài (UI)"""
        self.is_running = False
        # If your stt_manager has a method to force an immediate stop, call it here
        # E.g., self.stt_manager.stop_listening()

# WIDGET CHÍNH
class TrueLiquidWidget(QWidget):
    CARD_WIDTH = 700
    CARD_HEIGHT = 66

    def __init__(self, pipeline=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.resize(self.CARD_WIDTH, self.CARD_HEIGHT)

        self.dragging = False
        self.is_expanded = True

        # --- State management variables ---
        self.is_listening = False
        self.has_text = False

        self.card = TrueLiquidGlassCard(self)
        self.card.setGeometry(0, 0, self.CARD_WIDTH, self.CARD_HEIGHT)

        self.input = FocusLineEdit(self.card)
        self.input.setPlaceholderText("Ask anything...")
        font = QFont("Segoe UI", 12)
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 102)
        self.input.setFont(font)
        self.input.setFixedHeight(46)

        self.input.focusChanged.connect(self.card.set_focused)
        self.input.returnPressed.connect(self.on_enter_pressed)
        self.input.textChanged.connect(self.on_text_changed)

        self.path_root = Path(__file__).resolve().parent.parent.parent

        self.voice_button = QPushButton(self.card)
        self.voice_button.setIcon(QIcon(f"{self.path_root}/images/icons8-ai-generated-sound-48-2.png"))
        self.voice_button.setIconSize(QSize(24, 24))

        self.voice_button.clicked.connect(self.on_voice_button_clicked)
        self.voice_button.setFixedSize(46, 46)
        self.voice_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.mic_input = None

        self.add_btn_container = QWidget(self.card)
        self.add_btn_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.add_btn_container.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.add_btn_container.setAutoFillBackground(False)
        self.add_btn_container.setStyleSheet("background: transparent; border: none;")
        self.add_btn_container.setFixedSize(54, 54)

        self.add_btn_strands = Strands(
            self.add_btn_container,
            colors=["#00F2FE", "#4FACFE", "#7C3AED", "#FF0844"],
            count=4, speed=0.6, intensity=0.95, opacity=1.0, glow=0.8,
            thickness=1.4, saturation=1.8, glass=True, glass_size=1.0,
            refraction=1.5, dispersion=1.2, corner_radius=23, taper=0.5, scale=1.3
        )
        self.add_btn_strands.setGeometry(0, 0, 54, 54)
        self.add_btn_container.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.add_btn_strands.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)

        self.effect_opacity = QGraphicsOpacityEffect(self.add_btn_container)
        self.add_btn_strands.opacity = 0.0
        self.add_btn_container.hide()

        self.fade_anim_add_btn = QVariantAnimation(self)
        self.fade_anim_add_btn.setDuration(350)
        self.fade_anim_add_btn.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.fade_anim_add_btn.valueChanged.connect(self._on_fade_step)

        self.add_button = QPushButton("✕", self.card)
        self.add_button.setFixedSize(46, 46)
        self.add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_button.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.add_button.raise_()
        self.add_button.clicked.connect(self.toggle_mode)

        self._layout_inner_widgets(self.CARD_WIDTH)

        self.expanded_rect = QRect(0, 0, self.CARD_WIDTH, self.CARD_HEIGHT)
        self.collapsed_rect = QRect(0, 0, self.CARD_HEIGHT, self.CARD_HEIGHT)

        self.card_anim = QPropertyAnimation(self.card, b"geometry")
        self.card_anim.setDuration(450)
        self.card_anim.setEasingCurve(QEasingCurve.Type.InOutExpo)
        self.card_anim.finished.connect(self._on_anim_finished)
        self.card_anim.valueChanged.connect(self._on_anim_step)

        self.setStyleSheet("""
            QLineEdit { background: transparent; border: none; color: #ffffff; padding-left: 8px; selection-color: #ffffff; selection-background-color: rgba(255, 255, 255, 60);}
            QLineEdit::placeholder { color: rgba(255, 255, 255, 130); }
            QPushButton { background: transparent; border: none; color: rgba(255,255,255,180); font-size: 24px; border-radius: 23px; font-weight: bold;}
            QPushButton:hover { background: rgba(255, 255, 255, 30); color: #ffffff; }
        """)

        for widget in (self.card, self.input, self.add_btn_container, self.add_button, self.voice_button):
            widget.installEventFilter(self)

        self._start_entrance_animation()

        self.pipe_line = pipeline
        self.thread_pipeline = None
        self.worker_pipeline = None

        self.mic_thread = None
        self.mic_worker = None

    # ================= User Interface (UI) Handling Functions =================

    def _on_fade_step(self, value):
        self.add_btn_strands.opacity = value
        self.add_btn_strands.update()

    def _layout_inner_widgets(self, width):
        self.add_button.move(10, 10)
        self.add_btn_container.move(6, 5)
        self.voice_button.move(width - 56, 10)
        input_w = max(0, width - 134)
        self.input.move(68, 10)
        self.input.setFixedSize(input_w, 46)

    def toggle_mode(self):
        if self.is_expanded:
            self.add_button.setText("")
            self.input.hide()
            self.voice_button.hide()
            self.add_btn_container.show()
            self.fade_anim_add_btn.stop()
            self.fade_anim_add_btn.setStartValue(self.add_btn_strands.opacity)
            self.fade_anim_add_btn.setEndValue(1.0)
            try:
                self.fade_anim_add_btn.finished.disconnect()
            except TypeError:
                pass
            self.fade_anim_add_btn.start()

            self.card_anim.setStartValue(self.card.geometry())
            self.card_anim.setEndValue(self.collapsed_rect)
            self.card_anim.start()
            self.is_expanded = False
        else:
            self.add_button.setText("✕")
            self.fade_anim_add_btn.stop()
            self.fade_anim_add_btn.setStartValue(self.add_btn_strands.opacity)
            self.fade_anim_add_btn.setEndValue(0.0)
            try:
                self.fade_anim_add_btn.finished.disconnect()
            except TypeError:
                pass
            self.fade_anim_add_btn.finished.connect(self._on_fade_out_finished)
            self.fade_anim_add_btn.start()

            self.card_anim.setStartValue(self.card.geometry())
            self.card_anim.setEndValue(self.expanded_rect)
            self.card_anim.start()
            self.is_expanded = True

    def _on_fade_out_finished(self):
        if self.is_expanded and self.add_btn_strands.opacity == 0.0:
            self.add_btn_container.hide()

    def _on_anim_finished(self):
        if self.is_expanded:
            self.input.show()
            self.voice_button.show()
            self.input.setFocus()
        else:
            self.add_btn_container.show()

    def _on_anim_step(self, rect):
        self._layout_inner_widgets(rect.width())
        self.setFixedSize(rect.width(), rect.height())
        self.update()

    def _start_entrance_animation(self):
        self.setWindowOpacity(0.0)
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(400)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_anim.start()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
            global_pos = event.globalPosition().toPoint()
            card_pos = self.card.mapFromGlobal(global_pos)
            self.card.update_light_target(QPointF(card_pos))

            is_inside = self.card.rect().contains(card_pos)
            if is_inside != self.card.is_hovered:
                self.card.set_hovered(is_inside)
        elif event.type() == QEvent.Type.Leave and obj == self.card:
            self.card.set_hovered(False)

        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            local_pos = self.mapFromGlobal(event.globalPosition().toPoint())
            if self.card.geometry().contains(local_pos):
                self.drag_start_position = event.globalPosition().toPoint()
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                self.dragging = False
                return False

        elif event.type() == QEvent.Type.MouseMove and (event.buttons() & Qt.MouseButton.LeftButton):
            current_pos = event.globalPosition().toPoint()
            if not self.dragging and (current_pos - self.drag_start_position).manhattanLength() >= 5:
                self.dragging = True
            if self.dragging:
                self.move(current_pos - self.drag_position)
                return True
        elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            was_dragging = self.dragging
            self.dragging = False
            return was_dragging
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

    # ================= TEXT & BUTTON LOGIC =================

    def on_text_changed(self, text):
        self.has_text = bool(text.strip())

        if self.is_listening:
            return

        if self.has_text:
            pixmap = QPixmap(f"{self.path_root}/images/icons8-next-24.png").transformed(QTransform().rotate(-90))
            self.voice_button.setIcon(QIcon(pixmap))
        else:
            self.voice_button.setIcon(QIcon(f"{self.path_root}/images/icons8-ai-generated-sound-48-2.png"))

    def on_enter_pressed(self):
        user_input = self.input.text().strip()
        if user_input:
            self.start_pipeline(user_input)

    def on_voice_button_clicked(self):
        if self.has_text and not self.is_listening:
            user_input = self.input.text().strip()
            self.start_pipeline(user_input)
        else:
            self.toggle_mic()

    # ================= LOGIC MIC (CONTINUOUS) =================

    def toggle_mic(self):
        if self.is_listening:
            self.stop_mic()
        else:
            self.start_mic()

    def start_mic(self):
        if self.mic_input is None:
            self.mic_input = STTManager()
            AppContext.tts = self.mic_input

        self.is_listening = True
        self.update_mic_ui_state()

        self.mic_thread = QThread()

        self.mic_worker = MicWorker(self.mic_input)
        self.mic_worker.moveToThread(self.mic_thread)

        self.mic_thread.started.connect(self.mic_worker.run)
        self.mic_worker.text_received.connect(self.on_mic_text_received)
        self.mic_worker.error.connect(self.on_mic_error)

        self.mic_worker.finished.connect(self.mic_thread.quit)
        self.mic_thread.finished.connect(self.mic_worker.deleteLater)

        self.mic_thread.start()

    def stop_mic(self):
        """Completely turn off continuous listening mode."""
        self.is_listening = False

        # Phát tín hiệu dừng vòng lặp bên trong Worker
        if self.mic_worker:
            self.mic_worker.stop()

        self.update_mic_ui_state()
        self.on_text_changed(self.input.text())

    def on_mic_text_received(self, text):
        """Whenever the microphone picks up a sentence -> immediately feed it into the pipeline."""
        if text:
            self.start_pipeline(text)

    def on_mic_error(self, error):
        print(f"[Mic] Error: {error}")

    def update_mic_ui_state(self):
        if self.is_listening:
            self.voice_button.setIcon(QIcon(f"{self.path_root}/images/icons8-ai-generated-sound-48-2.png"))
            self.voice_button.setStyleSheet("""
                QPushButton { background: #3d5dff; border: none; color: rgba(255,255,255,180); font-size: 24px; border-radius: 23px; font-weight: bold;}
                QPushButton:hover { background: rgba(100, 125, 255, 30); color: #ffffff; }
            """)
            self.input.clear()
            self.input.setPlaceholderText("Đang nghe liên tục... (Nhấn để tắt)")
            self.input.setEnabled(False)
        else:
            self.voice_button.setStyleSheet("""
                QPushButton { background: transparent; border: none; color: rgba(255,255,255,180); font-size: 24px; border-radius: 23px; font-weight: bold;}
                QPushButton:hover { background: rgba(255, 255, 255, 30); color: #ffffff; }
            """)
            self.input.setPlaceholderText("Hỏi bất kỳ điều gì...")
            self.input.setEnabled(True)

    # ================= LOGIC PIPELINE (AI COMMUNICATION) =================

    def start_pipeline(self, user_input):
        if not user_input:
            return

        if self.thread_pipeline is not None and self.thread_pipeline.isRunning():
            print("[WARN] The pipeline is busy processing the previous request, skipping this one.:", user_input)
            return

        print(f"[User] {user_input}")

        self.input.clear()
        if self.is_listening:
            self.input.setPlaceholderText(f"Processing: {user_input[:20]}...")
        else:
            self.input.setEnabled(False)

        self.thread_pipeline = QThread()
        self.worker_pipeline = PipelineWorker(self.pipe_line, user_input)
        self.worker_pipeline.moveToThread(self.thread_pipeline)

        self.thread_pipeline.started.connect(self.worker_pipeline.run)
        self.worker_pipeline.finished.connect(self.on_pipeline_finished)
        self.worker_pipeline.error.connect(self.on_pipeline_error)

        self.worker_pipeline.finished.connect(self.thread_pipeline.quit)
        self.worker_pipeline.error.connect(self.thread_pipeline.quit)
        self.thread_pipeline.finished.connect(self.worker_pipeline.deleteLater)
        self.thread_pipeline.finished.connect(self.on_thread_pipeline_finished)

        self.thread_pipeline.start()

    def on_pipeline_finished(self, result):
        if not self.is_listening:
            self.input.setEnabled(True)
            self.input.setFocus()
        else:
            self.input.setPlaceholderText("Listening continuously... (Tap to stop)")

    def on_pipeline_error(self, error):
        print("[PIP] Error:", repr(error))
        if not self.is_listening:
            self.input.setEnabled(True)
            self.input.setFocus()
        else:
            self.input.setPlaceholderText("Listening continuously... (Tap to stop)")

    def on_thread_pipeline_finished(self):
        print("[PIP] AI processing thread terminated.")


class PipelineWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, pipe_line, user_input):
        super().__init__()
        self.pipe_line = pipe_line
        self.user_input = user_input

    def run(self):
        try:
            result = asyncio.run_coroutine_threadsafe(self.pipe_line.main(self.user_input), loop=self.pipe_line.loop)
            self.finished.emit(result)
        except Exception as e:
            print("[Worker Error]", repr(e))
            self.error.emit(e)


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    QSurfaceFormat.setDefaultFormat(_default_surface_format())
    app = QApplication(sys.argv)
    window = TrueLiquidWidget()
    window.show()
    sys.exit(app.exec())