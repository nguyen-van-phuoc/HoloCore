import sys
import numpy as np
from OpenGL import GL as gl
from PyQt6.QtGui import QSurfaceFormat, QColor
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QElapsedTimer
from PyQt6.QtWidgets import QApplication, QVBoxLayout

def _default_surface_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setAlphaBufferSize(8)
    fmt.setSamples(4)
    return fmt

class Strands(QOpenGLWidget):
    MAX_STRANDS = 12
    MAX_COLORS = 8

    VERT_SRC = """#version 330 core
        layout(location = 0) in vec2 position;
        void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        }
    """

    FRAG_SRC = """#version 330 core

        uniform float uTime;
        uniform vec2 uResolution;
        uniform vec3 uColors[8];
        uniform int uColorCount;
        uniform int uStrandCount;
        uniform float uSpeed;
        uniform float uAmplitude;
        uniform float uWaviness;
        uniform float uThickness;
        uniform float uGlow;
        uniform float uTaper;
        uniform float uSpread;
        uniform float uHueShift;
        uniform float uIntensity;
        uniform float uOpacity;
        uniform float uScale;
        uniform float uSaturation;

        out vec4 fragColor;

        const float PI = 3.14159265;

        vec3 spectrum(float t) {
            return 0.5 + 0.5 * cos(2.0 * PI * (t + vec3(0.00, 0.33, 0.67)));
        }

        vec3 samplePalette(float t) {
            t = fract(t);
            float scaled = t * float(uColorCount);
            int idx = int(floor(scaled));
            float blend = fract(scaled);
            int nextIdx = idx + 1;
            if (nextIdx >= uColorCount) nextIdx = 0;
                return mix(uColors[idx], uColors[nextIdx], blend);
        }

        vec3 strandColor(float t) {
            if (uColorCount > 0) return samplePalette(t);
            return spectrum(t);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
            uv /= max(uScale, 0.0001);

            float e = 0.06 + uIntensity * 0.94;
            float env = pow(max(cos(uv.x * PI * 1.3), 0.0), uTaper);

            vec3 col = vec3(0.0);

            for (int i = 0; i < 12; i++) {
                if (i >= uStrandCount) break;

                float fi = float(i);
                float ph = fi * 1.7 * uSpread;
                float freq = (2.0 + fi * 0.35) * uWaviness;
                float spd = 1.4 + fi * 1.2;

                float tt = uTime * uSpeed;
                float w = sin(uv.x * freq + tt * spd + ph) * 0.60
                + sin(uv.x * freq * 1.1 - tt * spd * 0.7 + ph * 1.7) * 0.40;

                float amp = (0.1 + 0.02 * e) * env * uAmplitude;
                float y = w * amp;

                float d = abs(uv.y - y);
                float thick = (0.001 + 0.05 * e) * (0.35 + env) * uThickness;
                float g = thick / (d + thick * 0.45);
                g = g * g;

                float h = fi / float(uStrandCount) + uv.x * 0.30 + uTime * 0.04 + uHueShift;
                col += strandColor(h) * g * env;
            }

            col *= 0.45 + 0.7 * e;
            col = 1.0 - exp(-col * uGlow);

            float gray = dot(col, vec3(0.2126, 0.7152, 0.0722));
            col = max(mix(vec3(gray), col, uSaturation), 0.0);

            float lum = max(max(col.r, col.g), col.b);
            float alpha = clamp(lum, 0.0, 1.0) * uOpacity;

            fragColor = vec4(col * uOpacity, alpha);
        }
    """

    GLASS_FRAG_SRC = """#version 330 core

        uniform sampler2D uScene;
        uniform vec2 uResolution;
        uniform float uRadius;
        uniform float uRefraction;
        uniform float uDispersion;
        uniform float uCornerRadius;

        out vec4 fragColor;

        vec2 toUv(vec2 p) {
            return p * (uResolution.y / uResolution) + 0.5;
        }
        
        float roundedBoxAlpha(vec2 fragCoord, vec2 resolution, float radius) {
            if (radius <= 0.0) return 1.0;
            vec2 center = 0.5 * resolution;
            vec2 halfSize = 0.5 * resolution - vec2(radius);
            vec2 p = abs(fragCoord - center) - halfSize;
            float dist = length(max(p, 0.0)) + min(max(p.x, p.y), 0.0) - radius;
            return 1.0 - smoothstep(-0.5, 1.5, dist);
        }

        void main() {
            vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
            float d = length(p);
            float r = uRadius;

    float edge = fwidth(d) * 1.5;
    float mask = 1.0 - smoothstep(r - edge, r + edge, d);
    if (mask <= 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    // sphere height: 0 at the rim, 1 at the center
    float z = sqrt(max(r * r - d * d, 0.0)) / r;
    float nd = d / r; // 0 at the center, 1 at the rim

    // refraction is confined to a narrow band near the rim; the rest stays undistorted
    vec2 dir = d > 0.0 ? p / d : vec2(0.0);
    float lens = smoothstep(0.85, 1.0, nd) * pow(nd, 6.0);
    vec2 offset = -dir * lens * uRefraction * 0.15;
    vec2 disp = -dir * lens * uDispersion * 0.012;

    vec3 light;
    light.r = texture(uScene, toUv(p + offset - disp)).r;
    light.g = texture(uScene, toUv(p + offset)).g;
    light.b = texture(uScene, toUv(p + offset + disp)).b;

    // neutral fresnel rim (no color tint so the glass stays clear)
    float fres = pow(1.0 - z, 3.0);
    vec3 rim = vec3(1.0) * fres * 0.18;

    // specular highlight from the upper-left
    vec2 lightDir = normalize(vec2(-0.55, 0.6));
    float spec = pow(max(dot(p / max(r, 1e-4), lightDir), 0.0), 6.0);
    spec *= smoothstep(r, r * 0.55, d);

    vec3 emissive = light + rim + vec3(spec) * 0.4;
    float emissiveA = clamp(max(max(emissive.r, emissive.g), emissive.b), 0.0, 1.0);

    // almost clear glass body: only a faint neutral darkening, mostly near the rim
    float bodyA = 0.05 + fres * 0.05;

    // composite emissive light over the clear body (premultiplied)
    float outA = emissiveA + bodyA * (1.0 - emissiveA);
    vec3 outRGB = emissive;

    outRGB *= mask;
    outA *= mask;
    
    float cornerAlpha = roundedBoxAlpha(gl_FragCoord.xy, uResolution, uCornerRadius);
    outRGB *= cornerAlpha;
    outA *= cornerAlpha;

    fragColor = vec4(outRGB, outA);
}
"""

    def __init__(
        self,
        parent=None,
        *,
        colors=None,
        count: int = 3,
        speed: float = 0.5,
        amplitude: float = 1.0,
        waviness: float = 1.0,
        thickness: float = 0.7,
        glow: float = 2.6,
        taper: float = 3.0,
        spread: float = 1.0,
        hue_shift: float = 0.0,
        intensity: float = 0.6,
        saturation: float = 1.5,
        opacity: float = 1.0,
        scale: float = 1.5,
        glass: bool = False,
        refraction: float = 1.0,
        dispersion: float = 1.0,
        glass_size: float = 1.0,
        corner_radius: int = 20,
        fps: int = 60,
    ):
        super().__init__(parent)
        self.setFormat(_default_surface_format())

        self.colors = list(colors) if colors else ["#FF4242", "#7C3AED", "#06B6D4", "#EAB308"]
        self.count = count
        self.speed = speed
        self.amplitude = amplitude
        self.waviness = waviness
        self.thickness = thickness
        self.glow = glow
        self.taper = taper
        self.spread = spread
        self.hue_shift = hue_shift
        self.intensity = intensity
        self.saturation = saturation
        self.opacity = opacity
        self.scale = scale
        self.glass = glass
        self.refraction = refraction
        self.dispersion = dispersion
        self.glass_size = glass_size
        self.corner_radius = corner_radius

        # Widget-level transparency. Also make sure the surface format
        # (whichever one is active when this widget's context is created)
        # has an alpha channel — see _default_surface_format().
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAutoFillBackground(False)

        # GL object handles, created lazily in initializeGL/resizeGL.
        self._program = None
        self._glass_program = None
        self._uniforms = {}
        self._glass_uniforms = {}
        self._vao = None
        self._vbo = None
        self._fbo = None
        self._fbo_texture = None
        self._fbo_w = 1
        self._fbo_h = 1

        self._elapsed = QElapsedTimer()
        self._elapsed.start()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)
        self._timer.start(max(1, round(1000 / fps)))

    # ------------------------------------------------------------------
    # Qt / OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        # QOpenGLWidget doesn't expose its own "about to be destroyed"
        # signal — the underlying QOpenGLContext does, and it's only
        # available once the context has actually been created.
        self.context().aboutToBeDestroyed.connect(self._cleanup_gl)

        self._program = self._compile_program(self.VERT_SRC, self.FRAG_SRC)
        self._glass_program = self._compile_program(self.VERT_SRC, self.GLASS_FRAG_SRC)

        self._uniforms = self._locate_uniforms(
            self._program,
            [
                "uTime", "uResolution", "uColors", "uColorCount", "uStrandCount",
                "uSpeed", "uAmplitude", "uWaviness", "uThickness", "uGlow",
                "uTaper", "uSpread", "uHueShift", "uIntensity", "uOpacity",
                "uScale", "uSaturation",
            ],
        )
        self._glass_uniforms = self._locate_uniforms(
            self._glass_program,
            ["uScene", "uResolution", "uRadius", "uRefraction", "uDispersion", "uCornerRadius"],
            # <--- Thêm uCornerRadius
        )

        # Full-screen triangle: (-1,-1), (3,-1), (-1,3) — covers the
        # viewport without needing a second triangle or UVs.
        positions = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, positions, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

    def resizeGL(self, w: int, h: int):
        ratio = self.devicePixelRatioF()
        w_phys = max(int(self.width() * ratio), 1)
        h_phys = max(int(self.height() * ratio), 1)
        self._resize_fbo(w_phys, h_phys)


    def paintGL(self):
        # [THÊM DÒNG NÀY]: Đảm bảo Qt không lấy màu đen đục làm nền mỗi khi vẽ lại
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

        t = self._elapsed.elapsed() / 1000.0
        strand_count = max(1, min(round(self.count), self.MAX_STRANDS))
        palette = self._build_palette()
        color_count = min(len(self.colors), self.MAX_COLORS)

        ratio = self.devicePixelRatioF()
        target_w = max(int(self.width() * ratio), 1)
        target_h = max(int(self.height() * ratio), 1)

        if self.glass and (self._fbo_w != target_w or self._fbo_h != target_h):
            self._resize_fbo(target_w, target_h)

        if self.glass:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
            gl.glViewport(0, 0, self._fbo_w, self._fbo_h)
            # [SỬA]: Thêm bit xoá độ sâu để an toàn tuyệt đối
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self._draw_strands(t, strand_count, palette, color_count, self._fbo_w, self._fbo_h)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            gl.glViewport(0, 0, target_w, target_h)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self._draw_glass(target_w, target_h)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            gl.glViewport(0, 0, target_w, target_h)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self._draw_strands(t, strand_count, palette, color_count, target_w, target_h)

    def _cleanup_gl(self):
        self._timer.stop()
        self.makeCurrent()
        try:
            if self._vao is not None:
                gl.glDeleteVertexArrays(1, [self._vao])
            if self._vbo is not None:
                gl.glDeleteBuffers(1, [self._vbo])
            if self._fbo_texture is not None:
                gl.glDeleteTextures(1, [self._fbo_texture])
            if self._fbo is not None:
                gl.glDeleteFramebuffers(1, [self._fbo])
            if self._program is not None:
                gl.glDeleteProgram(self._program)
            if self._glass_program is not None:
                gl.glDeleteProgram(self._glass_program)
        finally:
            self.doneCurrent()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_strands(self, t, strand_count, palette, color_count, w, h):
        gl.glUseProgram(self._program)
        u = self._uniforms
        gl.glUniform1f(u["uTime"], t)
        gl.glUniform2f(u["uResolution"], float(w), float(h))
        gl.glUniform3fv(u["uColors"], self.MAX_COLORS, palette)
        gl.glUniform1i(u["uColorCount"], color_count)
        gl.glUniform1i(u["uStrandCount"], strand_count)
        gl.glUniform1f(u["uSpeed"], self.speed)
        gl.glUniform1f(u["uAmplitude"], self.amplitude)
        gl.glUniform1f(u["uWaviness"], self.waviness)
        gl.glUniform1f(u["uThickness"], self.thickness)
        gl.glUniform1f(u["uGlow"], self.glow)
        gl.glUniform1f(u["uTaper"], self.taper)
        gl.glUniform1f(u["uSpread"], self.spread)
        gl.glUniform1f(u["uHueShift"], self.hue_shift)
        gl.glUniform1f(u["uIntensity"], self.intensity)
        gl.glUniform1f(u["uOpacity"], self.opacity)
        gl.glUniform1f(u["uScale"], self.scale)
        gl.glUniform1f(u["uSaturation"], self.saturation)

        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

    def _draw_glass(self, w, h):
        gl.glUseProgram(self._glass_program)
        u = self._glass_uniforms

        ratio = self.devicePixelRatioF()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fbo_texture)
        gl.glUniform1i(u["uScene"], 0)
        gl.glUniform2f(u["uResolution"], float(w), float(h))
        gl.glUniform1f(u["uRadius"], 0.46 * self.glass_size)
        gl.glUniform1f(u["uRefraction"], self.refraction)
        gl.glUniform1f(u["uDispersion"], self.dispersion)

        # Truyền bán kính bo góc (đã nhân với tỷ lệ DPI của màn hình)
        gl.glUniform1f(u["uCornerRadius"], float(self.corner_radius * ratio))  # <--- THÊM DÒNG NÀY

        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

    def _build_palette(self) -> np.ndarray:
        """Pad self.colors (hex strings) out to MAX_COLORS vec3s, repeating
        the last color — mirrors the original component's buildPalette()."""
        source = self.colors if self.colors else ["#ffffff"]
        out = np.empty((self.MAX_COLORS, 3), dtype=np.float32)
        for i in range(self.MAX_COLORS):
            hex_color = source[i] if i < len(source) else source[-1]
            c = QColor(hex_color)
            out[i] = (c.redF(), c.greenF(), c.blueF())
        return out

    # ------------------------------------------------------------------
    # GL resource (re)creation
    # ------------------------------------------------------------------

    def _resize_fbo(self, w: int, h: int):
        if self._fbo_texture is not None:
            gl.glDeleteTextures(1, [self._fbo_texture])
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, [self._fbo])

        self._fbo_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fbo_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fbo_texture, 0
        )
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"Strands: framebuffer incomplete (status=0x{status:x})", file=sys.stderr)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        self._fbo_w, self._fbo_h = w, h

    @staticmethod
    def _compile_shader(src: str, shader_type: int) -> int:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(shader).decode()
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Strands: shader compile error:\n{log}")
        return shader

    @classmethod
    def _compile_program(cls, vert_src: str, frag_src: str) -> int:
        vs = cls._compile_shader(vert_src, gl.GL_VERTEX_SHADER)
        fs = cls._compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode()
            gl.glDeleteProgram(program)
            raise RuntimeError(f"Strands: program link error:\n{log}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    @staticmethod
    def _locate_uniforms(program: int, names: list) -> dict:
        gl.glUseProgram(program)
        return {name: gl.glGetUniformLocation(program, name) for name in names}