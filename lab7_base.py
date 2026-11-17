import math


class Matrix4:
    def __init__(self, data=None):
        if data is None:
            self.m = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        else:
            self.m = [row[:] for row in data]

    @staticmethod
    def identity():
        return Matrix4()

    @staticmethod
    def translation(tx, ty, tz):
        return Matrix4([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def scale(sx, sy, sz):
        return Matrix4([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def rotation_x(angle_deg):
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        return Matrix4([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def rotation_y(angle_deg):
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        return Matrix4([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def rotation_z(angle_deg):
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        return Matrix4([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def reflection(plane):
        if plane == 'xy':
            return Matrix4.scale(1, 1, -1)
        if plane == 'yz':
            return Matrix4.scale(-1, 1, 1)
        if plane == 'xz':
            return Matrix4.scale(1, -1, 1)
        raise ValueError(plane)

    @staticmethod
    def rotation_around_axis_line(point, direction, angle_deg):
        x, y, z = direction
        length = math.sqrt(x * x + y * y + z * z)
        if length < 1e-12:
            raise ValueError('Направляющий вектор не может быть нулевым')
        l, m, n = x / length, y / length, z / length
        c = math.cos(math.radians(angle_deg))
        s = math.sin(math.radians(angle_deg))
        a, b, cc = point
        rot = Matrix4([
            [l * l + c * (1 - l * l), l * (1 - c) * m - n * s, l * (1 - c) * n + m * s, 0],
            [l * (1 - c) * m + n * s, m * m + c * (1 - m * m), m * (1 - c) * n - l * s, 0],
            [l * (1 - c) * n - m * s, m * (1 - c) * n + l * s, n * n + c * (1 - n * n), 0],
            [0, 0, 0, 1],
        ])
        return Matrix4.translation(a, b, cc) * rot * Matrix4.translation(-a, -b, -cc)

    def __mul__(self, other):
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                result.m[i][j] = sum(self.m[i][k] * other.m[k][j] for k in range(4))
        return result

    def apply(self, p):
        x, y, z = p
        nx = self.m[0][0] * x + self.m[0][1] * y + self.m[0][2] * z + self.m[0][3]
        ny = self.m[1][0] * x + self.m[1][1] * y + self.m[1][2] * z + self.m[1][3]
        nz = self.m[2][0] * x + self.m[2][1] * y + self.m[2][2] * z + self.m[2][3]
        nw = self.m[3][0] * x + self.m[3][1] * y + self.m[3][2] * z + self.m[3][3]
        if abs(nw) > 1e-12 and nw != 1:
            nx, ny, nz = nx / nw, ny / nw, nz / nw
        return (nx, ny, nz)


class Point:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def coords(self):
        return (self.x, self.y, self.z)


class Polygon:
    __slots__ = ('indices',)

    def __init__(self, indices):
        self.indices = list(indices)


class Polyhedron:
    def __init__(self, points=None, polygons=None, name='Модель'):
        self.points = points if points else []
        self.polygons = polygons if polygons else []
        self.name = name


def build_tetrahedron():
    a = 1.0
    vertices = [
        Point(a, a, a),
        Point(a, -a, -a),
        Point(-a, a, -a),
        Point(-a, -a, a),
    ]
    faces = [
        Polygon([0, 1, 2]),
        Polygon([0, 3, 1]),
        Polygon([0, 2, 3]),
        Polygon([1, 3, 2]),
    ]
    return Polyhedron(vertices, faces, 'Тетраэдр')


def build_hexahedron():
    a = 1.0
    vertices = [
        Point(-a, -a, -a), Point(a, -a, -a), Point(a, a, -a), Point(-a, a, -a),
        Point(-a, -a, a), Point(a, -a, a), Point(a, a, a), Point(-a, a, a),
    ]
    faces = [
        Polygon([0, 1, 2, 3]),
        Polygon([4, 5, 6, 7]),
        Polygon([0, 1, 5, 4]),
        Polygon([2, 3, 7, 6]),
        Polygon([1, 2, 6, 5]),
        Polygon([0, 3, 7, 4]),
    ]
    return Polyhedron(vertices, faces, 'Гексаэдр (куб)')


def build_octahedron():
    a = 1.0
    vertices = [
        Point(a, 0, 0), Point(-a, 0, 0),
        Point(0, a, 0), Point(0, -a, 0),
        Point(0, 0, a), Point(0, 0, -a),
    ]
    faces = [
        Polygon([0, 2, 4]), Polygon([2, 1, 4]),
        Polygon([1, 3, 4]), Polygon([3, 0, 4]),
        Polygon([2, 0, 5]), Polygon([1, 2, 5]),
        Polygon([3, 1, 5]), Polygon([0, 3, 5]),
    ]
    return Polyhedron(vertices, faces, 'Октаэдр')


def build_icosahedron():
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    faces = [
        Polygon([0, 11, 5]), Polygon([0, 5, 1]), Polygon([0, 1, 7]), Polygon([0, 7, 10]), Polygon([0, 10, 11]),
        Polygon([1, 5, 9]), Polygon([5, 11, 4]), Polygon([11, 10, 2]), Polygon([10, 7, 6]), Polygon([7, 1, 8]),
        Polygon([3, 9, 4]), Polygon([3, 4, 2]), Polygon([3, 2, 6]), Polygon([3, 6, 8]), Polygon([3, 8, 9]),
        Polygon([4, 9, 5]), Polygon([2, 4, 11]), Polygon([6, 2, 10]), Polygon([8, 6, 7]), Polygon([9, 8, 1]),
    ]
    return Polyhedron([Point(*v) for v in verts], faces, 'Икосаэдр (бонус)')


def build_dodecahedron():
    phi = (1 + math.sqrt(5)) / 2
    inv_phi = 1 / phi
    verts = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, inv_phi, phi), (0, inv_phi, -phi), (0, -inv_phi, phi), (0, -inv_phi, -phi),
        (inv_phi, phi, 0), (inv_phi, -phi, 0), (-inv_phi, phi, 0), (-inv_phi, -phi, 0),
        (phi, 0, inv_phi), (phi, 0, -inv_phi), (-phi, 0, inv_phi), (-phi, 0, -inv_phi),
    ]
    faces = [
        Polygon([0, 8, 10, 2, 16]), Polygon([0, 16, 17, 1, 12]), Polygon([0, 12, 14, 4, 8]),
        Polygon([1, 17, 3, 11, 9]), Polygon([1, 9, 5, 14, 12]), Polygon([2, 10, 6, 15, 13]),
        Polygon([2, 13, 3, 17, 16]), Polygon([3, 13, 15, 7, 11]), Polygon([4, 14, 5, 19, 18]),
        Polygon([4, 18, 6, 10, 8]), Polygon([5, 9, 11, 7, 19]), Polygon([6, 18, 19, 7, 15]),
    ]
    return Polyhedron([Point(*v) for v in verts], faces, 'Додекаэдр (бонус)')


BUILDERS = {
    'Тетраэдр': build_tetrahedron,
    'Гексаэдр (куб)': build_hexahedron,
    'Октаэдр': build_octahedron,
    'Икосаэдр (бонус)': build_icosahedron,
    'Додекаэдр (бонус)': build_dodecahedron,
}


def _circle(radius, count, center_y=0.0, center_radius=0.0):
    points = []
    for i in range(count + 1):
        angle = 2 * math.pi * i / count
        points.append((center_radius + radius * math.cos(angle), center_y + radius * math.sin(angle), 0.0))
    return points


def _semicircle(count=16):
    points = []
    for i in range(1, count):
        angle = -math.pi / 2 + math.pi * i / count
        points.append((math.cos(angle), math.sin(angle), 0.0))
    return points


PROFILE_PRESETS = {
    'Ваза': [
        (0.46, -1.15, 0.0),
        (0.72, -0.85, 0.0),
        (0.90, -0.35, 0.0),
        (0.74, 0.15, 0.0),
        (0.40, 0.58, 0.0),
        (0.36, 0.92, 0.0),
        (0.64, 1.16, 0.0),
    ],
    'Цилиндр': [(1.0, -1.0, 0.0), (1.0, 1.0, 0.0)],
    'Конус': [(1.0, -1.0, 0.0), (0.08, 1.0, 0.0)],
    'Сфера (профиль)': _semicircle(),
    'Тор (профиль)': _circle(0.42, 20, 0.0, 1.15),
}


SURFACE_FUNCTIONS = {
    'z = sin(x) · cos(y)': lambda x, y: math.sin(x) * math.cos(y),
    'z = cos(x² + y²)': lambda x, y: math.cos(x * x + y * y),
    'z = exp(-(x² + y²))': lambda x, y: math.exp(-(x * x + y * y)),
}


def build_surface_of_revolution(profile, axis='y', divisions=24):
    if axis != 'y':
        raise ValueError('Поверхность вращения строится вокруг оси Y.')
    count = int(divisions)
    if count < 3:
        raise ValueError('Количество разбиений: от 3.')
    rings = []
    points = []
    for x, y, z in profile:
        radius = math.hypot(float(x), float(z))
        height = float(y)
        if radius < 1e-8:
            rings.append(('pole', len(points)))
            points.append(Point(0.0, height, 0.0))
            continue
        rings.append(('ring', len(points)))
        for i in range(count):
            angle = 2 * math.pi * i / count
            points.append(Point(radius * math.cos(angle), height, radius * math.sin(angle)))
    polygons = []
    for (kind0, index0), (kind1, index1) in zip(rings, rings[1:]):
        if kind0 == 'pole' and kind1 == 'pole':
            continue
        if kind0 == 'pole':
            for i in range(count):
                polygons.append(Polygon([index0, index1 + i, index1 + (i + 1) % count]))
        elif kind1 == 'pole':
            for i in range(count):
                polygons.append(Polygon([index1, index0 + (i + 1) % count, index0 + i]))
        else:
            for i in range(count):
                j = (i + 1) % count
                polygons.append(Polygon([index0 + i, index1 + i, index1 + j, index0 + j]))
    return Polyhedron(points, polygons, 'Поверхность вращения')


def build_graph_surface(function, x_min, x_max, y_min, y_max, x_steps, y_steps, name):
    if x_steps < 1 or y_steps < 1:
        raise ValueError('Сетка поверхности должна содержать хотя бы одну ячейку.')
    points = []
    for i in range(x_steps + 1):
        x = x_min + (x_max - x_min) * i / x_steps
        for j in range(y_steps + 1):
            y = y_min + (y_max - y_min) * j / y_steps
            points.append(Point(x, y, function(x, y)))

    def index(i, j):
        return i * (y_steps + 1) + j

    polygons = []
    for i in range(x_steps):
        for j in range(y_steps):
            polygons.append(Polygon([
                index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)]))
    return Polyhedron(points, polygons, name)


def save_obj(poly, path):
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(f'o {poly.name}\n')
        for point in poly.points:
            stream.write(f'v {point.x:.12g} {point.y:.12g} {point.z:.12g}\n')
        for face in poly.polygons:
            stream.write('f ' + ' '.join(str(i + 1) for i in face.indices) + '\n')
