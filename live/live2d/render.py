import json
import ctypes, platform
from pathlib import Path
import live2d.v3 as live2d

if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()


from PyQt6.QtGui import QCursor
from PyQt6.QtCore import Qt, QTimer
from live2d.v3 import StandardParams
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from live.live2d.lip_sync_state import LipSyncState


class Live2DWidget(QOpenGLWidget):
    def __init__(self, lip_sync: LipSyncState):
        super().__init__()

        self.model = None
        self.path_model = str(Path(__file__).resolve().parent.parent /
                           "model" / "live2d" / "hiyori_en" / "hiyori_pro" /
                           "runtime" / "hiyori_pro_t11.model3.json")

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(200, 600)

        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.smooth_factor = 0.02

        self.motion_gropus = self.load_motion_groups(self.path_model)

        self.lip_sync = lip_sync
        self.lip_sync_n = 1.15

    def initializeGL(self):
        live2d.glInit()
        live2d.init()
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.path_model)
        self.model.Resize(self.width(), self.height())

        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right() - self.width() - 20
        y = screen.bottom() - self.height() - 20
        self.move(x, y)

        # timer render ~60fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def tick(self):
       try:
           global_postion = QCursor.pos()
           local_position = self.mapFromGlobal(global_postion)
           target_x = local_position.x()
           target_y = local_position.y()
           self.smooth_x += (target_x - self.smooth_x) * self.smooth_factor
           self.smooth_y += (target_y - self.smooth_y) * self.smooth_factor
           self.model.Drag(self.smooth_x, self.smooth_y)
           self.update()
       except KeyboardInterrupt as e:
           pass

    def resizeGL(self, w, h):
        if self.model:
            self.model.Resize(w, h)

    def paintGL(self):
        live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
        self.model.Update()
        rms = self.lip_sync.get_current_rms()
        self.model.SetParameterValue(StandardParams.ParamMouthOpenY, rms * self.lip_sync_n, 1.0)
        self.model.Draw()

    def load_motion_groups(self, model3_json_path: str, idle_priority: int = 1, action_priority: int = 3) -> dict:
        with open(model3_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            motions = data.get("FileReferences", {}).get("Motions", {})
        return {group: (idle_priority if group.lower() == "idle" else action_priority) for group in motions}

    def play_animation(self, name: str, no: int | None = None) -> bool:
        try:
            if name not in self.motion_gropus:
                print(f"[WARN] Group '{name}' does not exist. Valid.: {list(self.motion_gropus)}")
                return False
            priority = self.motion_gropus[name]
            if no is not None:
                self.model.StartMotion(name, no, priority)
            else:
                self.model.StartRandomMotion(group=name, priority=priority)
            return True
        except Exception as e:
            print(f"LIVE] {e}")

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        postion = event.position()
        x = postion.x()
        y = postion.y()
        print(f"[INFO] Clicked at: ({x}, {y})")
        if self.model.HitPart(x, y):
            print(f"[INFO] Hit part: ({x}, {y})")
            self.play_animation("Tap@Body", no=0)

    def info(self):
        print([x for x in dir(self.model) if
               any(k in x.lower() for k in ["drawable", "vertex", "index", "mesh", "part"])])
        print([x for x in dir(self.model) if "Model" in x or "model" in x])






