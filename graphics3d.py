from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from lab7_base import (BUILDERS, PROFILE_PRESETS, Matrix4, Point, Polygon, Polyhedron,
                       build_surface_of_revolution)

EPS = 1e-10
BASE_DIR = Path(__file__).resolve().parent


def unit(vector):
    v = np.asarray(vector, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < EPS:
        raise ValueError('Вектор должен быть конечным и ненулевым.')
    return v / length


def face_normal(points):
    p = np.asarray(points, dtype=float)
    n = np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0)
    return unit(n)


def orient_faces(vertices, faces, extras=None):
    extras = [] if extras is None else extras
    edges = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edges[min(a, b), max(a, b)].append((i, 1 if a < b else -1))
    if any(len(items) > 2 for items in edges.values()):
        raise ValueError('Немногообразная сетка: у ребра больше двух граней.')
    adjacency = defaultdict(list)
    for items in edges.values():
        if len(items) == 2:
            (a, da), (b, db) = items
            adjacency[a].append((b, -da * db))
            adjacency[b].append((a, -da * db))
    signs, components = {}, []
    for start in range(len(faces)):
        if start in signs:
            continue
        signs[start] = 1
        queue, component = deque([start]), []
        while queue:
            a = queue.popleft()
            component.append(a)
            for b, relation in adjacency[a]:
                expected = signs[a] * relation
                if b in signs and signs[b] != expected:
                    raise ValueError('Невозможно согласовать обход граней сетки.')
                if b not in signs:
                    signs[b] = expected
                    queue.append(b)
        components.append(component)

    def turned(value, sign):
        if value is None:
            return None
        value = np.array(value, copy=True)
        return value[::-1] if sign < 0 else value

    result = [f[:] if signs[i] == 1 else f[::-1] for i, f in enumerate(faces)]
    extra_out = [[turned(group[i], signs[i]) for i in range(len(faces))] for group in extras]
    closed_count = 0
    for component in components:
        closed = all(len(edges[min(a, b), max(a, b)]) == 2
                     for i in component
                     for a, b in zip(result[i], result[i][1:] + result[i][:1]))
        if not closed:
            continue
        closed_count += 1
        volume6 = 0.0
        origin = vertices[result[component[0]][0]]
        for i in component:
            p = vertices[result[i]] - origin
            volume6 += sum(np.dot(p[0], np.cross(p[j], p[j + 1]))
                           for j in range(1, len(p) - 1))
        if volume6 < 0:
            for i in component:
                result[i].reverse()
                for group in extra_out:
                    if group[i] is not None:
                        group[i] = group[i][::-1]
    return result, closed_count == len(components), extra_out


def triangulate(vertices, face):
    if len(face) == 3:
        return [tuple(face)]
    p = vertices[face]
    normal = face_normal(p)
    xy = np.delete(p, int(np.argmax(np.abs(normal))), axis=1)

    def cross(a, b, c):
        u, v = b - a, c - a
        return u[0] * v[1] - u[1] * v[0]

    area = sum(xy[i, 0] * xy[(i + 1) % len(xy), 1]
               - xy[(i + 1) % len(xy), 0] * xy[i, 1] for i in range(len(xy)))
    sign = 1 if area > 0 else -1
    remaining, triangles = list(range(len(face))), []
    while len(remaining) > 3:
        for j, b in enumerate(remaining):
            a, c = remaining[j - 1], remaining[(j + 1) % len(remaining)]
            turn = sign * cross(xy[a], xy[b], xy[c])
            if abs(turn) <= EPS:
                remaining.pop(j)
                break
            if turn < 0:
                continue
            if any(all(sign * cross(xy[u], xy[v], xy[k]) >= -EPS
                       for u, v in ((a, b), (b, c), (c, a)))
                   for k in remaining if k not in (a, b, c)):
                continue
            triangles.append((face[a], face[b], face[c]))
            remaining.pop(j)
            break
        else:
            raise ValueError('Не удалось разбить грань: проверьте самопересечения.')
    if len(remaining) == 3:
        triangles.append(tuple(face[i] for i in remaining))
    return triangles


def compute_vertex_normals(vertices, faces, face_normals=None):
    """Нормаль вершины — нормированная сумма нормалей смежных граней с весом площади."""
    vertices = np.asarray(vertices, dtype=float)
    if face_normals is None:
        face_normals = [face_normal(vertices[face]) for face in faces]
    accumulated = np.zeros_like(vertices)
    for face, normal in zip(faces, face_normals):
        index = np.asarray(face)
        points = vertices[index]
        area = np.linalg.norm(np.cross(points[1:-1] - points[0], points[2:] - points[0]).sum(axis=0))
        accumulated[index] += normal * (area if area > EPS else 1.0)
    length = np.linalg.norm(accumulated, axis=1, keepdims=True)
    valid = length[:, 0] > EPS
    accumulated[valid] /= length[valid]
    accumulated[~valid] = (0.0, 0.0, 1.0)
    return accumulated


def generate_uvs(count):
    if count == 3:
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    if count == 4:
        return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) - np.pi / 2
    return np.column_stack((0.5 + 0.5 * np.cos(angles), 0.5 + 0.5 * np.sin(angles)))


def bilinear_interpolate(values, a, b, c):
    """Билинейная интерполяция атрибутов вершин треугольника.

    На треугольнике она совпадает с барицентрической: a * v0 + b * v1 + c * v2.
    """
    values = np.asarray(values, dtype=float)
    a, b, c = (np.asarray(item, dtype=float) for item in (a, b, c))
    return a[..., None] * values[0] + b[..., None] * values[1] + c[..., None] * values[2]


def interpolated_normal(normals, a, b, c):
    normal = bilinear_interpolate(normals, a, b, c)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.divide(normal, np.maximum(length, EPS))


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: list
    name: str = 'Модель'
    imported_normals: object = None
    imported_uvs: object = None
    normals: np.ndarray = field(init=False)
    triangles: list = field(init=False)
    closed: bool = field(init=False)
    corner_normals: list = field(init=False)
    corner_uvs: list = field(init=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or not len(self.vertices) or not np.isfinite(self.vertices).all()):
            raise ValueError('Некорректные координаты вершин.')
        clean, clean_normals, clean_uvs = [], [], []
        for index, face in enumerate(self.faces):
            raw = list(face)
            compact = list(dict.fromkeys(raw))
            if any(i < 0 or i >= len(self.vertices) for i in compact):
                raise ValueError('Индекс грани за пределами массива вершин.')
            if len(compact) < 3:
                continue
            try:
                face_normal(self.vertices[compact])
            except ValueError:
                continue
            usable = compact == raw
            clean.append(compact)
            clean_normals.append(self._corner_attr(self.imported_normals, index, len(compact)) if usable else None)
            clean_uvs.append(self._corner_attr(self.imported_uvs, index, len(compact)) if usable else None)
        if not clean:
            raise ValueError('В модели нет невырожденных граней.')
        self.faces, self.closed, oriented = orient_faces(
            self.vertices, clean, [clean_normals, clean_uvs])
        clean_normals, clean_uvs = oriented
        self.normals = np.array([face_normal(self.vertices[face]) for face in self.faces])
        self.triangles = [triangulate(self.vertices, face) for face in self.faces]
        vertex_normals = compute_vertex_normals(self.vertices, self.faces, self.normals)
        corners = []
        for face, custom, normal in zip(self.faces, clean_normals, self.normals):
            if custom is None:
                corners.append(vertex_normals[np.asarray(face)])
                continue
            custom = np.asarray(custom, dtype=float)
            length = np.linalg.norm(custom, axis=1, keepdims=True)
            custom = np.divide(custom, np.maximum(length, EPS))
            if float(np.dot(custom.mean(axis=0), normal)) < 0:
                custom = -custom
            corners.append(custom)
        self.corner_normals = corners
        self.corner_uvs = []
        for face, custom in zip(self.faces, clean_uvs):
            if custom is None:
                self.corner_uvs.append(generate_uvs(len(face)))
            else:
                self.corner_uvs.append(np.asarray(custom, dtype=float)[:, :2])

    @staticmethod
    def _corner_attr(source, index, count):
        if source is None or index >= len(source) or source[index] is None:
            return None
        attr = np.asarray(source[index], dtype=float)
        if len(attr) != count:
            return None
        return attr

    @classmethod
    def from_polyhedron(cls, poly):
        return cls([p.coords() for p in poly.points],
                   [p.indices for p in poly.polygons], getattr(poly, 'name', 'Модель'))

    def normalized(self):
        center = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) / 2
        span = np.ptp(self.vertices, axis=0).max()
        return Mesh((self.vertices - center) * (2 / max(span, EPS)),
                    [list(face) for face in self.faces], self.name,
                    [np.array(normal, copy=True) for normal in self.corner_normals],
                    [np.array(uv, copy=True) for uv in self.corner_uvs])


def _obj_index(token, count, what):
    index = int(token)
    if index == 0 or not -count <= index <= count:
        raise ValueError(f'индекс {what} вне диапазона')
    return index - 1 if index > 0 else count + index


def _present(items):
    if items and any(item is not None for item in items):
        return items
    return None


def load_obj(path):
    vertices, normals, texcoords = [], [], []
    faces, face_normals, face_uvs = [], [], []
    with open(path, encoding='utf-8-sig') as stream:
        for number, raw in enumerate(stream, 1):
            parts = raw.split('#', 1)[0].split()
            if not parts:
                continue
            try:
                kind = parts[0]
                if kind == 'v':
                    if len(parts) < 4:
                        raise ValueError('нужны три координаты v')
                    vertex = [float(s) for s in parts[1:4]]
                    if len(parts) > 4:
                        w = float(parts[4])
                        if abs(w) < EPS:
                            raise ValueError('нулевая однородная координата')
                        vertex = [x / w for x in vertex]
                    vertices.append(vertex)
                elif kind == 'vn':
                    if len(parts) < 4:
                        raise ValueError('нужны три компоненты vn')
                    normals.append([float(s) for s in parts[1:4]])
                elif kind == 'vt':
                    if len(parts) < 3:
                        raise ValueError('нужны две координаты vt')
                    texcoords.append([float(s) for s in parts[1:3]])
                elif kind == 'f':
                    face, normal_ids, uv_ids = [], [], []
                    has_normal, has_uv = True, True
                    for token in parts[1:]:
                        bits = token.split('/')
                        face.append(_obj_index(bits[0], len(vertices), 'вершины'))
                        if len(bits) > 1 and bits[1]:
                            uv_ids.append(_obj_index(bits[1], len(texcoords), 'текстуры'))
                        else:
                            has_uv = False
                        if len(bits) > 2 and bits[2]:
                            normal_ids.append(_obj_index(bits[2], len(normals), 'нормали'))
                        else:
                            has_normal = False
                    if len(face) < 3:
                        raise ValueError('в грани меньше трёх вершин')
                    faces.append(face)
                    face_normals.append(np.array([normals[i] for i in normal_ids])
                                        if has_normal and len(normal_ids) == len(face) else None)
                    face_uvs.append(np.array([texcoords[i] for i in uv_ids])
                                    if has_uv and len(uv_ids) == len(face) else None)
            except (ValueError, OverflowError, IndexError) as error:
                raise ValueError(f'OBJ, строка {number}: {error}') from error
    if not vertices or not faces:
        raise ValueError('В OBJ нет вершин или граней.')
    return Mesh(vertices, faces, Path(path).stem, _present(face_normals), _present(face_uvs))


@dataclass
class SceneObject:
    mesh: Mesh
    color: tuple = (82, 167, 225)
    model: np.ndarray = field(default_factory=lambda: np.eye(4))

    def world_vertices(self):
        homogeneous = np.column_stack((self.mesh.vertices, np.ones(len(self.mesh.vertices))))
        return (homogeneous @ self.model.T)[:, :3]

    def apply(self, matrix, about_center=False):
        m = np.asarray(matrix.m if isinstance(matrix, Matrix4) else matrix, dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all() or abs(np.linalg.det(m[:3, :3])) < EPS:
            raise ValueError('Преобразование должно быть конечным и невырожденным; масштаб ≠ 0.')
        if about_center:
            center = self.world_vertices().mean(axis=0)
            t = np.eye(4)
            t[:3, 3] = center
            ti = np.eye(4)
            ti[:3, 3] = -center
            m = t @ m @ ti
        candidate = m @ self.model
        if not np.isfinite(candidate).all() or abs(np.linalg.det(candidate[:3, :3])) < EPS:
            raise ValueError('Слишком малый или большой масштаб.')
        self.model = candidate

    def world_normals(self):
        normals = self.mesh.normals @ np.linalg.inv(self.model[:3, :3])
        return normals / np.linalg.norm(normals, axis=1)[:, None]

    def world_corner_normals(self):
        inverse = np.linalg.inv(self.model[:3, :3])
        transformed = []
        for normals in self.mesh.corner_normals:
            world = normals @ inverse
            length = np.linalg.norm(world, axis=1, keepdims=True)
            transformed.append(world / np.maximum(length, EPS))
        return transformed


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([5., 3., 7.]))
    direction: np.ndarray = field(default_factory=lambda: np.array([-5., -3., -7.]))
    projection: str = 'perspective'
    fov: float = 50.0
    ortho_height: float = 6.0
    near: float = 0.1
    far: float = 100.0

    def view_matrix(self):
        forward = unit(self.direction)
        up = np.array([0., 1., 0.])
        if abs(np.dot(forward, up)) > .999:
            up = np.array([0., 0., 1.])
        right = unit(np.cross(forward, up))
        real_up = np.cross(right, forward)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([right, real_up, forward])
        matrix[:3, 3] = -matrix[:3, :3] @ np.asarray(self.position)
        return matrix

    def projection_matrix(self, aspect):
        n, f = self.near, self.far
        if not 0 < n < f or not math.isfinite(f) or not math.isfinite(aspect) or aspect <= 0:
            raise ValueError('Нужно 0 < near < far и положительное отношение сторон.')
        if self.projection == 'perspective':
            if not 1 <= self.fov <= 175:
                raise ValueError('Угол обзора должен быть от 1 до 175 градусов.')
            s = 1 / math.tan(math.radians(self.fov) / 2)
            return np.array([[s / aspect, 0, 0, 0], [0, s, 0, 0],
                             [0, 0, f / (f - n), -f * n / (f - n)], [0, 0, 1, 0]])
        if self.projection != 'orthographic' or not self.ortho_height > 0:
            raise ValueError('Некорректная параллельная проекция.')
        h = self.ortho_height
        return np.array([[2 / (h * aspect), 0, 0, 0], [0, 2 / h, 0, 0],
                         [0, 0, 1 / (f - n), -n / (f - n)], [0, 0, 0, 1]])

    def orbit(self, target, angle_deg):
        target = np.asarray(target, dtype=float)
        rotation = np.asarray(Matrix4.rotation_y(angle_deg).m)[:3, :3]
        self.position = target + rotation @ (np.asarray(self.position) - target)
        self.direction = target - self.position


@dataclass
class Light:
    position: np.ndarray
    color: tuple = (255, 255, 255)
    ambient: float = 0.18

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.color = tuple(int(c) for c in self.color)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError('Некорректное положение источника света.')
        if not 0 <= self.ambient <= 1 or any(c < 0 or c > 255 for c in self.color):
            raise ValueError('Цвет света или фоновая интенсивность заданы неверно.')


def shade_ndotl(normal, position, light_position):
    direction = np.asarray(light_position, dtype=float) - np.asarray(position, dtype=float)
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.divide(direction, np.maximum(length, EPS))
    return np.sum(np.asarray(normal, dtype=float) * direction, axis=-1)


def lambert_intensity(ndotl, ambient):
    """Диффузная модель Ламберта с фоновой составляющей: ka + (1 - ka) * max(N·L, 0)."""
    return ambient + (1.0 - ambient) * np.clip(ndotl, 0.0, 1.0)


def toon_intensity(ndotl, ambient):
    """Туншейдинг: N·L квантуется в четыре полосы яркости."""
    values = np.atleast_1d(np.clip(np.asarray(ndotl, dtype=float), 0.0, 1.0))
    span = 1.0 - ambient
    bands = np.array([ambient, ambient + span * 0.35, ambient + span * 0.68, 1.0])
    index = np.digitize(values, np.array([0.25, 0.50, 0.78]))
    result = bands[index]
    if np.asarray(ndotl).ndim == 0:
        return float(result[0])
    return result


def apply_light(color, light_color, intensity):
    rgb = np.asarray(color, dtype=float) * (np.asarray(light_color, dtype=float) / 255.0)
    return rgb * np.asarray(intensity, dtype=float)[..., None]


def lit_color(normal, position, color, light):
    intensity = lambert_intensity(float(np.clip(shade_ndotl(normal, position, light.position), 0.0, 1.0)),
                                  light.ambient)
    return apply_light(color, light.color, intensity)


def make_checker(size=256, cells=4):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // cells
    y, x = np.mgrid[0:size, 0:size]
    bright = ((x // cell) + (y // cell)) % 2 == 0
    image[bright] = (245, 236, 214)
    image[~bright] = (32, 86, 168)
    mark = max(8, size // 7)
    image[-mark:, :mark] = (214, 48, 48)
    image[-mark:, -mark:] = (42, 168, 74)
    image[:mark, :mark] = (46, 92, 214)
    return image


def make_circles(size=256):
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = (252, 246, 232)
    y, x = np.mgrid[0:size, 0:size]
    radius = np.sqrt((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
    ring = (radius / (size / 10)).astype(int) % 2 == 0
    image[ring] = (176, 64, 52)
    return image


def sample_texture(texture, uv):
    uv = np.asarray(uv, dtype=float)
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
    if len(uv) == 0:
        return np.zeros((0, 3))
    texture = np.asarray(texture, dtype=float)
    height, width = texture.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    x = u * width - 0.5
    y = (1.0 - v) * height - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    x0 = x0 % width
    y0 = y0 % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    return (c00 * (1 - tx) * (1 - ty) + c10 * tx * (1 - ty)
            + c01 * (1 - tx) * ty + c11 * tx * ty)


CLIP_PLANES = np.array([[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1],
                        [0, -1, 0, 1], [0, 0, 1, 0], [0, 0, -1, 1]])


def clip_polygon(points, attributes=None):
    polygon = [np.asarray(point, dtype=float) for point in points]
    attrs = None if attributes is None else [np.asarray(item, dtype=float) for item in attributes]
    for plane in CLIP_PLANES:
        if not polygon:
            break
        output, output_attrs = [], []
        a, da = polygon[-1], float(np.dot(polygon[-1], plane))
        ea = None if attrs is None else attrs[-1]
        for index, b in enumerate(polygon):
            db = float(np.dot(b, plane))
            eb = None if attrs is None else attrs[index]
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                output.append(a + (b - a) * t)
                if attrs is not None:
                    output_attrs.append(ea + (eb - ea) * t)
            if db >= 0:
                output.append(b)
                if attrs is not None:
                    output_attrs.append(eb)
            a, da, ea = b, db, eb
        polygon = output
        if attrs is not None:
            attrs = output_attrs
    if attributes is None:
        return polygon
    return polygon, attrs


@dataclass
class RenderResult:
    color: np.ndarray
    depth: np.ndarray
    stats: dict
    normal_lines: list

    def depth_image(self):
        finite = np.isfinite(self.depth)
        gray = np.zeros(self.depth.shape, dtype=np.uint8)
        if finite.any():
            low, high = self.depth[finite].min(), self.depth[finite].max()
            gray[finite] = (55 + 200 * (1 - (self.depth[finite] - low) / max(high - low, EPS))).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)


def project_to_pixels(point, camera, width, height):
    clip = camera.projection_matrix(width / height) @ camera.view_matrix() @ np.r_[np.asarray(point, float), 1.0]
    if not np.isfinite(clip).all() or clip[3] <= EPS:
        return None
    ndc = clip[:3] / clip[3]
    if abs(ndc[0]) > 1.05 or abs(ndc[1]) > 1.05 or not -0.02 <= float(ndc[2]) <= 1.02:
        return None
    return np.array([(ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2])


class SoftwareRenderer:
    def __init__(self, width=800, height=600):
        self.width, self.height = width, height

    def rasterize_triangle(self, points, color, image, depth, use_z=True, attributes=None, shader=None):
        p = np.asarray(points, dtype=float)
        lo = np.maximum(np.ceil(p[:, :2].min(axis=0) - .5).astype(int), 0)
        hi = np.minimum(np.floor(p[:, :2].max(axis=0) - .5).astype(int),
                        [self.width - 1, self.height - 1])
        if np.any(lo > hi):
            return
        x0, y0, z0 = p[0]
        x1, y1, z1 = p[1]
        x2, y2, z2 = p[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < EPS:
            return
        y, x = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        x, y = x + .5, y + .5
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
        c = 1 - a - b
        z = a * z0 + b * z1 + c * z2
        region = np.s_[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        mask = (a >= -EPS) & (b >= -EPS) & (c >= -EPS) & (z >= -EPS) & (z <= 1 + EPS)
        if use_z:
            mask &= z < depth[region]
        if not np.any(mask):
            return
        if attributes is None:
            image[region][mask] = color
        else:
            samples = bilinear_interpolate(attributes, a[mask], b[mask], c[mask])
            shaded = samples if shader is None else shader(samples)
            image[region][mask] = np.clip(np.rint(shaded), 0, 255).astype(np.uint8)
        depth[region][mask] = z[mask]

    def fill_polygon(self, clip_points, image, depth, use_z, color=None, attributes=None, shader=None):
        if attributes is None:
            polygon = clip_polygon(clip_points)
            attrs = None
        else:
            polygon, attrs = clip_polygon(clip_points, attributes)
        if len(polygon) < 3 or any(point[3] <= EPS for point in polygon):
            return 0
        screen = self.to_screen(polygon)
        attr_array = None if attrs is None else np.asarray(attrs, dtype=float)
        count = 0
        for j in range(1, len(screen) - 1):
            self.rasterize_triangle(
                screen[[0, j, j + 1]], color, image, depth, use_z,
                None if attr_array is None else attr_array[[0, j, j + 1]], shader)
            count += 1
        return count

    def to_screen(self, clip):
        clip = np.asarray(clip)
        ndc = clip[:, :3] / clip[:, 3:4]
        return np.column_stack(((ndc[:, 0] + 1) * self.width / 2,
                                (1 - ndc[:, 1]) * self.height / 2, ndc[:, 2]))

    def draw_edge(self, a, b, image, depth, use_z):
        a, b = a.copy(), b.copy()
        for plane in CLIP_PLANES:
            da, db = np.dot(a, plane), np.dot(b, plane)
            if da < 0 and db < 0:
                return
            if (da >= 0) != (db >= 0):
                intersection = a + (b - a) * da / (da - db)
                if da < 0:
                    a = intersection
                else:
                    b = intersection
        if min(a[3], b[3]) <= EPS:
            return
        p, q = self.to_screen([a, b])
        steps = int(np.ceil(np.max(np.abs(q[:2] - p[:2])))) + 1
        samples = np.linspace(p, q, max(steps, 1))
        x = np.floor(samples[:, 0]).astype(int)
        y = np.floor(samples[:, 1]).astype(int)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, z = x[valid], y[valid], samples[valid, 2]
        if use_z:
            valid = z <= depth[y, x] + .0003
            x, y = x[valid], y[valid]
        image[y, x] = (27, 40, 59)

    def render(self, objects, camera, cull=True, use_z=True, edges=False, normals=False,
               shading='flat', light=None, texture=None, light_texture=True):
        if shading not in {'flat', 'gouraud', 'phong', 'texture'}:
            raise ValueError('Неизвестный режим закраски.')
        if light is None:
            light = Light(np.array([3.5, 4.5, 5.0]))
        texture = make_checker() if texture is None else np.asarray(texture, dtype=float)
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (239, 243, 248)
        depth = np.full((self.height, self.width), np.inf)
        vp = camera.projection_matrix(self.width / self.height) @ camera.view_matrix()
        stats = dict(faces=0, culled=0, triangles=0, pixels=0, shading=shading)
        normal_lines, edge_list = [], []
        view = -unit(camera.direction) if camera.projection == 'orthographic' else None
        for obj in objects:
            world = obj.world_vertices()
            clip = np.column_stack((world, np.ones(len(world)))) @ vp.T
            world_normals = obj.world_normals()
            corner_normals = obj.world_corner_normals() if shading in {'gouraud', 'phong', 'texture'} else None
            phong_shader = self._phong_shader(obj.color, light, camera, view)
            texture_shader = self._texture_shader(texture)
            for i, face in enumerate(obj.mesh.faces):
                stats['faces'] += 1
                center, normal = world[face].mean(axis=0), world_normals[i]
                to_eye = (np.asarray(camera.position) - center if camera.projection == 'perspective'
                          else -unit(camera.direction))
                front = np.dot(normal, to_eye) > EPS
                if cull and not front:
                    stats['culled'] += 1
                    continue
                lookup = {vertex: corner for corner, vertex in enumerate(face)}
                for triangle in obj.mesh.triangles[i]:
                    tri = list(triangle)
                    ids = [lookup[vertex] for vertex in triangle]
                    positions = world[tri]
                    if shading == 'flat':
                        rgb = np.clip(np.rint(lit_color(normal, positions.mean(axis=0), obj.color, light)),
                                      0, 255).astype(np.uint8)
                        stats['triangles'] += self.fill_polygon(clip[tri], image, depth, use_z, rgb)
                    elif shading == 'gouraud':
                        colors = np.stack([
                            lit_color(corner_normals[i][corner], positions[vertex], obj.color, light)
                            for vertex, corner in enumerate(ids)])
                        stats['triangles'] += self.fill_polygon(
                            clip[tri], image, depth, use_z, attributes=colors, shader=lambda samples: samples)
                    elif shading == 'phong':
                        attributes = np.concatenate([corner_normals[i][ids], positions], axis=1)
                        stats['triangles'] += self.fill_polygon(
                            clip[tri], image, depth, use_z, attributes=attributes, shader=phong_shader)
                    else:
                        if light_texture:
                            colors = np.stack([
                                lit_color(corner_normals[i][corner], positions[vertex], obj.color, light)
                                for vertex, corner in enumerate(ids)])
                        else:
                            colors = np.tile(np.asarray(obj.color, dtype=float), (3, 1))
                        attributes = np.concatenate([obj.mesh.corner_uvs[i][ids], colors], axis=1)
                        stats['triangles'] += self.fill_polygon(
                            clip[tri], image, depth, use_z, attributes=attributes, shader=texture_shader)
                if edges:
                    edge_list.extend((clip[a], clip[b]) for a, b in zip(face, face[1:] + face[:1]))
                if normals and (len(obj.mesh.faces) < 150 or i % 8 == 0):
                    line = np.column_stack((np.array([center, center + normal * .28]), np.ones(2))) @ vp.T
                    if all(np.all(CLIP_PLANES @ p >= 0) and p[3] > EPS for p in line):
                        normal_lines.append((self.to_screen(line)[:, :2], front))
        for a, b in edge_list:
            self.draw_edge(a, b, image, depth, use_z)
        stats['pixels'] = int(np.isfinite(depth).sum())
        return RenderResult(image, depth, stats, normal_lines)

    @staticmethod
    def _phong_shader(color, light, camera, constant_view):
        def shader(samples):
            # Нормаль уже билинейно интерполирована между вершинами; здесь её нормируют.
            normal = samples[:, 0:3]
            position = samples[:, 3:6]
            length = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = np.divide(normal, np.maximum(length, EPS))
            ndotl = np.clip(shade_ndotl(normal, position, light.position), 0.0, 1.0)
            rgb = apply_light(color, light.color, toon_intensity(ndotl, light.ambient))
            if constant_view is None:
                view = camera.position - position
                view = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), EPS)
            else:
                view = np.broadcast_to(constant_view, normal.shape)
            light_dir = light.position - position
            light_dir = light_dir / np.maximum(np.linalg.norm(light_dir, axis=1, keepdims=True), EPS)
            half = light_dir + view
            half = half / np.maximum(np.linalg.norm(half, axis=1, keepdims=True), EPS)
            gloss = np.clip(np.sum(normal * half, axis=1), 0.0, 1.0) >= 0.94
            if np.any(gloss):
                rgb = np.array(rgb, copy=True)
                rgb[gloss] = np.asarray(light.color, dtype=float) * 0.75 + rgb[gloss] * 0.25
            return rgb
        return shader

    @staticmethod
    def _texture_shader(texture):
        def shader(samples):
            return sample_texture(texture, samples[:, :2]) * samples[:, 2:5] / 255.0
        return shader


def make_mesh(name):
    if name == 'Куб (OBJ)':
        return load_obj(BASE_DIR / 'cube.obj')
    if name == 'Сфера (OBJ)':
        return load_obj(BASE_DIR / 'sphere.obj')
    if name == 'Ваза (ЛР7)':
        profile = [(0, -1.4, 0)] + list(PROFILE_PRESETS['Ваза']) + [(0, 1.3, 0)]
        poly = build_surface_of_revolution(profile, 'y', 24)
        poly.name = 'Ваза (ЛР7)'
        return Mesh.from_polyhedron(poly)
    return Mesh.from_polyhedron(BUILDERS[name]())


MODEL_NAMES = ['Ваза (ЛР7)', 'Куб (OBJ)', 'Сфера (OBJ)'] + list(BUILDERS)


def demo_scene():
    cube = SceneObject(make_mesh('Куб (OBJ)'), (70, 158, 228))
    cube.apply(Matrix4.rotation_y(28))
    cube.apply(Matrix4.translation(-.65, 0, 0))
    sphere = SceneObject(make_mesh('Сфера (OBJ)'), (237, 141, 65))
    sphere.apply(Matrix4.translation(.7, .15, .3))
    small = SceneObject(make_mesh('Октаэдр'), (98, 193, 144))
    small.apply(Matrix4.scale(.85, .85, .85))
    small.apply(Matrix4.translation(.15, -.7, 1.25))
    return [cube, sphere, small]


def save_obj(obj, path):
    vertices = obj.world_vertices()
    faces = obj.mesh.faces
    if np.linalg.det(obj.model[:3, :3]) < 0:
        faces = [f[::-1] for f in faces]
    poly = Polyhedron([Point(*p) for p in vertices], [Polygon(f) for f in faces], obj.mesh.name)
    from lab7_base import save_obj as base_save_obj
    base_save_obj(poly, str(path))
