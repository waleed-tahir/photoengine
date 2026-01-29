"""
GPU-Accelerated Viewport

OpenGL/Vulkan viewport for real-time image display with LUT preview.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    HAS_QT = True
except ImportError:
    HAS_QT = False

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

try:
    from OpenGL import GL
    HAS_PYOPENGL = True
except ImportError:
    HAS_PYOPENGL = False


if HAS_QT and HAS_OPENGL and HAS_PYOPENGL:
    
    class GPUViewport(QOpenGLWidget):
        """
        GPU-accelerated viewport for real-time image display.
        
        Uses OpenGL for hardware-accelerated rendering with:
        - Real-time LUT application
        - Zoom and pan
        - Color picker
        - Split view comparison
        """
        
        VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        out vec2 TexCoord;
        
        uniform mat4 transform;
        
        void main() {
            gl_Position = transform * vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        FRAGMENT_SHADER = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        uniform sampler2D imageTexture;
        uniform sampler3D lutTexture;
        uniform bool useLUT;
        uniform float lutSize;
        uniform float exposure;
        uniform float gamma;
        
        vec3 applyLUT(vec3 color) {
            // Scale to LUT coordinates
            vec3 coords = color * (lutSize - 1.0) / lutSize + 0.5 / lutSize;
            return texture(lutTexture, coords).rgb;
        }
        
        void main() {
            vec3 color = texture(imageTexture, TexCoord).rgb;
            
            // Exposure
            color *= pow(2.0, exposure);
            
            // LUT
            if (useLUT) {
                color = applyLUT(clamp(color, 0.0, 1.0));
            }
            
            // Gamma
            color = pow(color, vec3(1.0 / gamma));
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.image_texture = None
            self.lut_texture = None
            self.shader_program = None
            
            self.exposure = 0.0
            self.gamma = 2.2
            self.use_lut = False
            self.lut_size = 33
            
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            
            self.setMinimumSize(400, 300)
        
        def initializeGL(self):
            """Initialize OpenGL resources."""
            GL.glClearColor(0.1, 0.1, 0.1, 1.0)
            
            # Create shader program
            self.shader_program = QOpenGLShaderProgram()
            self.shader_program.addShaderFromSourceCode(
                QOpenGLShader.Vertex, self.VERTEX_SHADER)
            self.shader_program.addShaderFromSourceCode(
                QOpenGLShader.Fragment, self.FRAGMENT_SHADER)
            self.shader_program.link()
            
            # Create quad vertices
            self.vertices = np.array([
                # Positions        # Texture coords
                -1.0, -1.0, 0.0,   0.0, 0.0,
                 1.0, -1.0, 0.0,   1.0, 0.0,
                 1.0,  1.0, 0.0,   1.0, 1.0,
                -1.0,  1.0, 0.0,   0.0, 1.0,
            ], dtype=np.float32)
            
            self.indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            
            # Create VAO, VBO, EBO
            self.vao = GL.glGenVertexArrays(1)
            self.vbo = GL.glGenBuffers(1)
            self.ebo = GL.glGenBuffers(1)
            
            GL.glBindVertexArray(self.vao)
            
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes,
                           self.vertices, GL.GL_STATIC_DRAW)
            
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes,
                           self.indices, GL.GL_STATIC_DRAW)
            
            # Position attribute
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                    5 * 4, None)
            GL.glEnableVertexAttribArray(0)
            
            # Texture coord attribute
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE,
                                    5 * 4, GL.ctypes.c_void_p(3 * 4))
            GL.glEnableVertexAttribArray(1)
            
            GL.glBindVertexArray(0)
        
        def resizeGL(self, w, h):
            GL.glViewport(0, 0, w, h)
        
        def paintGL(self):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            
            if self.image_texture is None:
                return
            
            self.shader_program.bind()
            
            # Set uniforms
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0] = self.zoom
            transform[1, 1] = self.zoom
            transform[3, 0] = self.pan_x
            transform[3, 1] = self.pan_y
            
            self.shader_program.setUniformValue("transform", transform.flatten().tolist())
            self.shader_program.setUniformValue("exposure", self.exposure)
            self.shader_program.setUniformValue("gamma", self.gamma)
            self.shader_program.setUniformValue("useLUT", self.use_lut)
            self.shader_program.setUniformValue("lutSize", float(self.lut_size))
            
            # Bind textures
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_texture)
            self.shader_program.setUniformValue("imageTexture", 0)
            
            if self.lut_texture and self.use_lut:
                GL.glActiveTexture(GL.GL_TEXTURE1)
                GL.glBindTexture(GL.GL_TEXTURE_3D, self.lut_texture)
                self.shader_program.setUniformValue("lutTexture", 1)
            
            # Draw
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            
            self.shader_program.release()
        
        def set_image(self, image: np.ndarray):
            """Upload image to GPU."""
            self.makeCurrent()
            
            if self.image_texture is None:
                self.image_texture = GL.glGenTextures(1)
            
            h, w = image.shape[:2]
            image_float = image.astype(np.float32)
            
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, w, h, 0,
                           GL.GL_RGB, GL.GL_FLOAT, image_float)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            
            self.update()
        
        def set_lut(self, lut: np.ndarray):
            """Upload 3D LUT to GPU."""
            self.makeCurrent()
            
            if self.lut_texture is None:
                self.lut_texture = GL.glGenTextures(1)
            
            size = lut.shape[0]
            self.lut_size = size
            lut_float = lut.astype(np.float32)
            
            GL.glBindTexture(GL.GL_TEXTURE_3D, self.lut_texture)
            GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGB32F,
                           size, size, size, 0, GL.GL_RGB, GL.GL_FLOAT, lut_float)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
            
            self.use_lut = True
            self.update()
        
        def wheelEvent(self, event):
            """Handle zoom."""
            delta = event.angleDelta().y() / 120
            self.zoom *= 1.1 ** delta
            self.zoom = np.clip(self.zoom, 0.1, 10.0)
            self.update()
        
        def mouseMoveEvent(self, event):
            """Handle pan."""
            if event.buttons() & Qt.MiddleButton:
                dx = event.x() - self._last_mouse_x
                dy = event.y() - self._last_mouse_y
                self.pan_x += dx / self.width() * 2
                self.pan_y -= dy / self.height() * 2
                self.update()
            
            self._last_mouse_x = event.x()
            self._last_mouse_y = event.y()
        
        def mousePressEvent(self, event):
            self._last_mouse_x = event.x()
            self._last_mouse_y = event.y()

else:
    class GPUViewport:
        """Fallback when OpenGL not available."""
        pass
