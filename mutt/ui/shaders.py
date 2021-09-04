import moderngl
import numpy as np
from typing import Tuple

Color = Tuple[float, float, float, float]
Point2 = Tuple[float, float]

_SIGNAL_VERTEX_SHADER = '''
    #version 330

    uniform int Start;
    uniform float XScale;
    uniform mat4 MainRect;

    in int in_vert;

    void main() {
        float x = 2. * (Start + gl_VertexID) * XScale - 1.;
        float y = in_vert / 32768.;
        gl_Position = vec4(x, y, 0.0, 1.0) * MainRect;
    }
'''

_BASIC_FRAGMENT_SHADER = '''
    #version 330

    uniform vec4 Color;

    out vec4 f_color;

    void main() {
        f_color = Color;
    }
'''

_BASIC_VERTEX_SHADER = '''
    #version 330
    out int inst;
    void main() {
        inst = gl_InstanceID;
    }
'''

_LINE_GEOMETRY_SHADER = '''
    #version 330
    layout (points) in;
    layout (line_strip, max_vertices = 2) out;
    uniform mat4x2 LineMatrix;
    uniform int Segments;
    uniform mat4 MainRect;
    void main() {
        float start = float(2 * gl_PrimitiveIDIn) / Segments;
        gl_Position = vec2(start, 1) * LineMatrix * MainRect;
        EmitVertex();
        float end = float(2 * gl_PrimitiveIDIn + 1) / Segments;
        gl_Position = vec2(end, 1) * LineMatrix * MainRect;
        EmitVertex();
        EndPrimitive();
    }
'''

_GRID_GEOMETRY_SHADER = '''
    #version 330
    layout (points) in;
    layout (line_strip, max_vertices = 2) out;
    in int inst[1];
    uniform mat4 MainRect;
    uniform int HCount;
    uniform int VCount;
    uniform int Segments;
    void vline(float n, float m) {
        float x = 0.;
        if (HCount != 0)
            x = 2 * n / HCount - 1;
        float y1 = 4 * m / Segments - 1;
        float y2 = (4 * m + 2) / Segments - 1;
        gl_Position = vec4(x, y1, 0, 1) * MainRect;
        EmitVertex();
        gl_Position = vec4(x, y2, 0, 1) * MainRect;
        EmitVertex();
        EndPrimitive();
    }
    void hline(float n, float m) {
        float y = 0;
        if (VCount != 0)
            y = 2 * n / VCount - 1;
        float x1 = 4 * m / Segments - 1;
        float x2 = (4 * m + 2) / Segments - 1;
        gl_Position = vec4(x1, y, 0, 1) * MainRect;
        EmitVertex();
        gl_Position = vec4(x2, y, 0, 1) * MainRect;
        EmitVertex();
        EndPrimitive();
    }
    void main() {
        if (inst[0] <= HCount)
            vline(inst[0], gl_PrimitiveIDIn);
        else
            hline(inst[0] - HCount - 1, gl_PrimitiveIDIn);
    }
'''

_TEXTURE_VERTEX_SHADER = '''
    #version 330

    uniform mat4 MainRect;
    uniform mat2 TexScale;
    uniform int TexOffset;
    uniform int Height;

    in vec2 in_vert;
    in vec2 in_texcoord;

    out vec2 v_texcoord;

    void main() {
        v_texcoord = (in_texcoord - vec2(0, float(TexOffset) / Height))
                     * TexScale;
        gl_Position = vec4(in_vert[0], in_vert[1], 0.0, 1.0) * MainRect;
    }
'''

_TEXTURE_FRAGMENT_SHADER = '''
    #version 330

    uniform sampler2D Texture;
    uniform int Palette;

    in vec2 v_texcoord;
    out vec4 f_color;

    vec4 palette0(float v) {
        return vec4(v*v, v*v*v, v*(1-v)+v*v*v, 1.);
    }

    vec4 palette1(float v) {
        return vec4(v, v, v*v*v*v, v);
    }

    vec4 palette2(float v) {
        return vec4(v, 0, 0, v);
    }

    void main() {
        float val = texture(Texture, v_texcoord)[0];
        if (Palette == 2) {
            f_color = palette2(val);
        }
        else if (Palette == 1) {
            f_color = palette1(val);
        }
        else {
            f_color = palette0(val);
        }
        if (f_color.a < 0.1)
            discard;
    }
'''


class VertexArrayWrapper:

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program,
                 contents: list, main_rect: np.ndarray):
        self.vao = ctx.vertex_array(program, contents)
        self.program['MainRect'] = tuple(main_rect.flatten())
        self.mode = None

    @property
    def program(self) -> 'moderngl.Program':
        return self.vao.program

    def render(self, mode=None, vertices=-1, **kargs):
        r_mode = mode if mode is not None else self.mode
        self.vao.render(mode=r_mode, vertices=vertices, **kargs)


class SignalVA(VertexArrayWrapper):
    VERTEX_WIDTH = 2

    def __init__(self, ctx: moderngl.Context, main_rect: np.ndarray,
                 max_count: int):
        program = ctx.program(
            vertex_shader=_SIGNAL_VERTEX_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.buffer = ctx.buffer(
            reserve=max_count * self.VERTEX_WIDTH)
        super().__init__(ctx, program, [(self.buffer, 'i2', 'in_vert')],
                         main_rect)
        self.mode = moderngl.LINE_STRIP

    def render_data(self, points: np.ndarray, count: int, scale: int,
                    start: int, color: Color):
        self.buffer.orphan()
        self.buffer.write(points)
        self.program['Color'] = color
        self.program['XScale'] = 1 / (scale - 1)
        self.program['Start'] = start
        super().render(vertices=count)


class GridVA(VertexArrayWrapper):
    # Grid color
    GRID_COLOR = (0.4, 0.4, 0.4, 1.0)
    # Color for frame, center lines and ticks
    FRAME_COLOR = (0.8, 0.8, 0.8, 1.0)
    # Number of segments of grid lines
    GRID_SEGMENTS = 101

    def __init__(self, ctx: moderngl.Context, main_rect: np.ndarray,
                 horizontal_div: int, vertical_div: int):
        program = ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_GRID_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        super().__init__(ctx, program, [], main_rect)
        self.mode = moderngl.POINTS
        self.horizontal_div = horizontal_div
        self.vertical_div = vertical_div

    def render_fragment(self, hcount: int, vcount: int, segments: int,
                        color: Color, ticks: bool = False):
        self.program['HCount'] = hcount
        self.program['VCount'] = vcount
        self.program['Color'] = color
        self.program['Segments'] = segments
        segs = segments // 2 + 1 if not ticks else 1
        super().render(vertices=segs, instances=hcount + vcount + 2)
        # super().render(vertices=segs, instances=1)

    def render_grid(self):
        # Render basic grid
        self.render_fragment(self.horizontal_div, self.vertical_div,
                             self.GRID_SEGMENTS, self.GRID_COLOR)
        # Render central horizontal and vertical lines
        self.render_fragment(0, 0, self.GRID_SEGMENTS, self.FRAME_COLOR)

    def render_frame(self):
        # Render frame
        self.render_fragment(1, 1, 1, self.FRAME_COLOR)
        # Render ticks
        self.render_fragment(self.horizontal_div * 5, self.vertical_div * 5,
                             200, self.FRAME_COLOR, ticks=1)


class LineVA(VertexArrayWrapper):

    def __init__(self, ctx: moderngl.Context, main_rect: np.ndarray):
        program = ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        super().__init__(ctx, program, [], main_rect)
        self.mode = moderngl.POINTS

    def render_line(self, start: Point2, end: Point2, color: Color,
                    segments: int):
        self.program['LineMatrix'] = (end[0] - start[0], start[0],
                                      end[1] - start[1], start[1],
                                      0,                 0,
                                      0,                 1)
        self.program['Color'] = color
        self.program['Segments'] = segments
        super().render(vertices=segments // 2 + 1)

    def render_hline(self, y: float, color: Color, segments: int):
        self.render_line((-1, y), (1, y), color, segments)

    def render_vline(self, x: float, color: Color, segments: int):
        self.render_line((x, -1), (x, 1), color, segments)


class TextureVA(VertexArrayWrapper):

    def __init__(self, ctx: moderngl.Context, main_rect: np.ndarray,
                 width: int, height: int,
                 palette: int = 0):
        program = ctx.program(
            vertex_shader=_TEXTURE_VERTEX_SHADER,
            fragment_shader=_TEXTURE_FRAGMENT_SHADER)
        self.width = width
        self.height = height
        self.buffer = ctx.buffer(
            np.array([
                -1.0, -1.0,  0, 0,  # lower left
                -1.0,  1.0,  0, 1,  # upper left
                1.0, -1.0,  1, 0,  # lower right
                1.0,  1.0,  1, 1,  # upper right
            ], dtype=np.float32))
        super().__init__(ctx, program,
                         [(self.buffer, '2f4 2f4', 'in_vert', 'in_texcoord')],
                         main_rect)
        self.program['Height'] = height
        self.program['TexScale'] = (1, 0, 0, 1)
        self.program['Palette'] = palette
        self.mode = moderngl.TRIANGLE_STRIP
        self.texture = ctx.texture((width, height), 1, dtype='f4')
        # self.texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.texture.swizzle = 'R001'
        self.texture_buffer = ctx.buffer(reserve=width * height * 4)
        self.position = 0

    def render(self):
        self.texture.use(location=0)
        super().render()

    def write_data(self, data: np.ndarray):
        self.texture_buffer.write(data, offset=(
                self.height - self.position - 1)
                * self.width * 4)
        self.texture.write(self.texture_buffer)
        self.program['TexOffset'] = self.position
        self.position = (self.position + 1) % self.height

    def render_data(self, data: np.ndarray):
        self.write_data(data)
        self.render()

    def set_scale(self, scale_x: float):
        self.program['TexScale'] = (1 / scale_x, 0, 0, 1)

    def clear(self):
        self.texture_buffer.clear()
