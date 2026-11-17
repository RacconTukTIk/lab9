import math
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from graphics3d import (BASE_DIR, MODEL_NAMES, Camera, Light, Matrix4, Mesh, SceneObject,
                        SoftwareRenderer, demo_scene, load_obj, make_checker, make_circles,
                        make_mesh, project_to_pixels, save_obj, unit)
from lab7_base import (PROFILE_PRESETS, SURFACE_FUNCTIONS, Polygon, build_graph_surface,
                       build_surface_of_revolution)


SHADING_MODES = {
    'Гуро (Ламберт)': 'gouraud',
    'Фонг (туншейдинг)': 'phong',
    'Текстура': 'texture',
    'Плоский Ламберт': 'flat',
}


class App:
    def __init__(self, root):
        self.root = root
        root.title('Лабораторная работа №9. Освещение и текстурирование')
        root.geometry('1200x760')
        root.minsize(1020, 660)
        self.camera = Camera()
        self.objects = []
        self.selected = 0
        self.target = np.zeros(3)
        self.dirty = True
        self.last_result = None
        self.photo = None
        self.last_time = time.perf_counter()
        self.light = Light(np.array([3.5, 4.5, 5.0]))
        self.checker_texture = make_checker()
        self.circle_texture = make_circles()
        self.custom_texture = None
        self.cull = tk.BooleanVar(value=True)
        self.zbuffer = tk.BooleanVar(value=True)
        self.edges = tk.BooleanVar(value=False)
        self.normals = tk.BooleanVar(value=False)
        self.depth_view = tk.BooleanVar(value=False)
        self.show_light = tk.BooleanVar(value=True)
        self.light_texture = tk.BooleanVar(value=True)
        self.animation = tk.StringVar(value='object')
        self.speed = tk.StringVar(value='25')
        self.projection = tk.StringVar(value='Перспективная')
        self.status = tk.StringVar()
        self.camera_status = tk.StringVar()
        self.model_info = tk.StringVar()
        self.mode = tk.StringVar(value='Z-буфер')
        self.shading = tk.StringVar(value='Гуро (Ламберт)')
        self.texture_name = tk.StringVar(value='Шахматка')
        self.normalize_obj = tk.BooleanVar(value=True)
        self.matrix_text = None
        self._build_ui()
        self.lighting_demo()
        root.bind('<space>', self.toggle_animation)
        root.bind('<Escape>', lambda _: self.animation.set('none'))
        root.after(30, self.tick)

    def invalidate(self, *_):
        self.dirty = True

    def guarded(self, action):
        def run():
            try:
                action()
                self.invalidate()
            except (ValueError, OSError, tk.TclError, np.linalg.LinAlgError) as error:
                messagebox.showerror('Некорректные данные', str(error), parent=self.root)
        return run

    @staticmethod
    def values(variables):
        values = [float(v.get().replace(',', '.')) for v in variables]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Введите конечные числа.')
        return values

    def entries(self, parent, label, defaults):
        ttk.Label(parent, text=label).pack(anchor='w', pady=(8, 2))
        row = ttk.Frame(parent)
        row.pack(fill='x')
        variables = []
        for default in defaults:
            variable = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=variable, width=8).pack(side='left', fill='x', expand=True, padx=2)
            variables.append(variable)
        return variables

    def button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=self.guarded(command))
        button.pack(fill='x', pady=3)
        return button

    def _build_ui(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label='Открыть OBJ…', command=self.guarded(self.open_obj))
        file_menu.add_command(label='Открыть текстуру…', command=self.guarded(self.open_texture))
        file_menu.add_command(label='Сохранить OBJ…', command=self.guarded(self.export_obj))
        file_menu.add_command(label='Сохранить изображение…', command=self.guarded(self.save_image))
        file_menu.add_separator()
        file_menu.add_checkbutton(label='Нормировать OBJ при загрузке', variable=self.normalize_obj)
        file_menu.add_separator()
        file_menu.add_command(label='Выход', command=self.root.destroy)
        menu.add_cascade(label='Файл', menu=file_menu)
        view_menu = tk.Menu(menu, tearoff=False)
        for text, variable in [('Рёбра', self.edges), ('Нормали', self.normals),
                               ('Глубина', self.depth_view), ('Источник света', self.show_light)]:
            view_menu.add_checkbutton(label=text, variable=variable, command=self.invalidate)
        view_menu.add_separator()
        view_menu.add_command(label='Матрица проекции', command=self.show_matrix)
        menu.add_cascade(label='Вид', menu=view_menu)
        scene_menu = tk.Menu(menu, tearoff=False)
        scene_menu.add_command(label='Вписать в кадр', command=self.guarded(self.fit_scene))
        scene_menu.add_command(label='Тетраэдр, куб и октаэдр', command=self.guarded(self.show_textured_polyhedra))
        scene_menu.add_command(label='Изменить порядок объектов', command=self.guarded(self.reverse_objects))
        menu.add_cascade(label='Сцена', menu=scene_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label='Управление', command=lambda: messagebox.showinfo(
            'Управление',
            'ЛКМ — поворот камеры\nКолесо — приближение\nПробел — пауза\nEsc — остановка\n\n'
            'Гуро интерполирует цвет модели Ламберта между вершинами.\n'
            'Фонг интерполирует нормали, нормирует их и считает туншейдинг.\n'
            'Текстура интерполирует координаты текстуры по грани.\n'
            'Нелицевые грани отбрасываются.', parent=self.root))
        menu.add_cascade(label='Справка', menu=help_menu)
        self.root.configure(menu=menu)
        outer = ttk.Panedwindow(self.root, orient='horizontal')
        outer.pack(fill='both', expand=True, padx=6, pady=6)
        left = ttk.Frame(outer, width=330)
        right = ttk.Frame(outer)
        outer.add(left, weight=0)
        outer.add(right, weight=1)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill='both', expand=True)
        self.light_tab = ttk.Frame(self.notebook, padding=10)
        self.scene_tab = ttk.Frame(self.notebook, padding=10)
        self.transform_tab = ttk.Frame(self.notebook, padding=10)
        self.camera_tab = ttk.Frame(self.notebook, padding=10)
        self.models_tab = ttk.Frame(self.notebook, padding=10)
        for frame, title in [(self.light_tab, 'Свет'), (self.scene_tab, 'Сцена'),
                             (self.transform_tab, 'Объект'), (self.camera_tab, 'Камера'),
                             (self.models_tab, 'Модели')]:
            self.notebook.add(frame, text=title)
        self._light_controls()
        self._scene_controls()
        self._transform_controls()
        self._camera_controls()
        self._models_controls()
        tools = ttk.Frame(right, padding=(8, 4))
        tools.pack(fill='x')
        shade = ttk.Combobox(tools, textvariable=self.shading, values=list(SHADING_MODES),
                             state='readonly', width=22)
        shade.pack(side='left', padx=(0, 8))
        shade.bind('<<ComboboxSelected>>', lambda _e: self.invalidate())
        mode_combo = ttk.Combobox(tools, textvariable=self.mode,
                                 values=['Отсечение граней', 'Z-буфер', 'Камера'], state='readonly', width=18)
        mode_combo.pack(side='left', padx=(0, 8))
        mode_combo.bind('<<ComboboxSelected>>', self.guarded_event(self.select_mode))
        for text, variable in [('Отсечение', self.cull), ('Z-буфер', self.zbuffer)]:
            ttk.Checkbutton(tools, text=text, variable=variable, command=self.invalidate).pack(side='left', padx=3)
        self.canvas = tk.Canvas(right, background='#eff3f8', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.invalidate)
        self.canvas.bind('<ButtonPress-1>', self.drag_start)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<MouseWheel>', self.zoom)
        self.canvas.bind('<Button-4>', lambda e: self.zoom(e, 1))
        self.canvas.bind('<Button-5>', lambda e: self.zoom(e, -1))
        ttk.Label(right, textvariable=self.status, padding=(8, 4)).pack(fill='x')
        footer = ttk.Frame(left, padding=(6, 8))
        footer.pack(fill='x')
        for text, value in [('Стоп', 'none'), ('Объект', 'object'), ('Камера', 'camera')]:
            ttk.Radiobutton(footer, text=text, variable=self.animation, value=value,
                            command=self.animation_changed).pack(side='left', padx=3)
        ttk.Label(footer, text='°/с:').pack(side='left', padx=(8, 2))
        ttk.Entry(footer, textvariable=self.speed, width=6).pack(side='left')

    def _light_controls(self):
        tab = self.light_tab
        ttk.Label(tab, text=(
            'Гуро — цвет Ламберта в вершинах и билинейная интерполяция.\n'
            'Фонг — интерполяция нормалей, нормализация, туншейдинг.\n'
            'Текстура — интерполяция UV по грани.'
        ), wraplength=300, justify='left').pack(anchor='w')
        ttk.Label(tab, text='Закраска').pack(anchor='w', pady=(8, 2))
        shade = ttk.Combobox(tab, textvariable=self.shading, values=list(SHADING_MODES), state='readonly')
        shade.pack(fill='x')
        shade.bind('<<ComboboxSelected>>', lambda _e: self.invalidate())
        self.light_pos = self.entries(tab, 'Положение источника x, y, z', (3.5, 4.5, 5))
        self.light_color = self.entries(tab, 'Цвет света R, G, B', (255, 255, 255))
        ttk.Label(tab, text='Фон 0…1').pack(anchor='w', pady=(8, 2))
        self.ambient = tk.StringVar(value='0.18')
        ttk.Entry(tab, textvariable=self.ambient).pack(fill='x')
        for variable in (*self.light_pos, *self.light_color, self.ambient):
            variable.trace_add('write', self.invalidate)
        ttk.Checkbutton(tab, text='Показывать источник', variable=self.show_light,
                        command=self.invalidate).pack(anchor='w', pady=(8, 0))
        ttk.Separator(tab).pack(fill='x', pady=10)
        self.object_color = self.entries(tab, 'Цвет объекта R, G, B', (222, 170, 120))
        self.button(tab, 'Применить цвет к объекту', self.apply_object_color)
        ttk.Separator(tab).pack(fill='x', pady=10)
        ttk.Label(tab, text='Текстура').pack(anchor='w')
        texture = ttk.Combobox(tab, textvariable=self.texture_name,
                               values=['Шахматка', 'Круги', 'Файл'], state='readonly')
        texture.pack(fill='x', pady=4)
        texture.bind('<<ComboboxSelected>>', lambda _e: self.invalidate())
        ttk.Checkbutton(tab, text='Освещать текстуру по Ламберту', variable=self.light_texture,
                        command=self.invalidate).pack(anchor='w')
        self.button(tab, 'Открыть текстуру…', self.open_texture)
        self.button(tab, 'Тетраэдр, куб, октаэдр', self.show_textured_polyhedra)

    def _scene_controls(self):
        tab = self.scene_tab
        ttk.Label(tab, text='Модель').pack(anchor='w')
        self.model_name = tk.StringVar(value='Сфера (OBJ)')
        ttk.Combobox(tab, textvariable=self.model_name, values=MODEL_NAMES, state='readonly', width=28).pack(fill='x', pady=4)
        row = ttk.Frame(tab)
        row.pack(fill='x')
        ttk.Button(row, text='Заменить сцену', command=self.guarded(lambda: self.add_model(True))).pack(
            side='left', fill='x', expand=True)
        ttk.Button(row, text='Добавить', command=self.guarded(self.add_model)).pack(side='left', fill='x', expand=True, padx=(4, 0))
        ttk.Label(tab, text='Объекты').pack(anchor='w', pady=(10, 3))
        self.object_list = tk.Listbox(tab, height=8, exportselection=False)
        self.object_list.pack(fill='x')
        self.object_list.bind('<<ListboxSelect>>', self.select_object)
        row = ttk.Frame(tab)
        row.pack(fill='x', pady=4)
        ttk.Button(row, text='Удалить', command=self.guarded(self.remove_object)).pack(side='left', expand=True, fill='x')
        ttk.Label(tab, textvariable=self.model_info).pack(anchor='w', pady=4)
        self.button(tab, 'Вписать в кадр', self.fit_scene)
        self.button(tab, 'Текстура на многогранниках', self.show_textured_polyhedra)

    def _models_controls(self):
        tab = self.models_tab
        ttk.Label(tab, text='Поверхность вращения').pack(anchor='w')
        row = ttk.Frame(tab)
        row.pack(fill='x', pady=3)
        self.profile = tk.StringVar(value='Ваза')
        self.divisions = tk.StringVar(value='24')
        ttk.Combobox(row, textvariable=self.profile, values=list(PROFILE_PRESETS), state='readonly', width=20).pack(side='left')
        ttk.Entry(row, textvariable=self.divisions, width=5).pack(side='left', padx=5)
        self.button(tab, 'Построить вокруг Y', self.build_revolution)
        ttk.Separator(tab).pack(fill='x', pady=12)
        ttk.Label(tab, text='Поверхность функции').pack(anchor='w')
        self.function = tk.StringVar(value='z = sin(x) · cos(y)')
        ttk.Combobox(tab, textvariable=self.function, values=list(SURFACE_FUNCTIONS), state='readonly').pack(fill='x', pady=3)
        self.button(tab, 'Построить на [−π, π]²', self.build_graph)

    def _transform_controls(self):
        tab = self.transform_tab
        self.translation = self.entries(tab, 'Смещение dx, dy, dz', (0.3, 0, 0))
        self.button(tab, 'Сместить', lambda: self.transform(Matrix4.translation(*self.values(self.translation))))
        self.scaling = self.entries(tab, 'Масштаб sx, sy, sz', (1.2, 1.2, 1.2))
        self.button(tab, 'Масштабировать', lambda: self.transform(Matrix4.scale(*self.values(self.scaling)), True))
        self.angle = self.entries(tab, 'Угол поворота, °', (30,))
        self.axis = tk.StringVar(value='y')
        ttk.Combobox(tab, textvariable=self.axis, values=['x', 'y', 'z'], state='readonly').pack(fill='x', pady=3)
        self.button(tab, 'Повернуть вокруг центра', self.rotate_selected)
        row = ttk.Frame(tab)
        row.pack(fill='x', pady=8)
        ttk.Label(row, text='Отражение:').pack(side='left')
        self.plane = tk.StringVar(value='xy')
        ttk.Combobox(row, textvariable=self.plane, values=['xy', 'xz', 'yz'], state='readonly', width=6).pack(side='left', padx=5)
        ttk.Button(row, text='Отразить', command=self.guarded(
            lambda: self.transform(Matrix4.reflection(self.plane.get()), True))).pack(side='left')
        self.button(tab, 'Сбросить преобразования', self.reset_object)
        extra = self.collapsible(tab, 'Произвольная ось')
        self.line_point = self.entries(extra, 'Точка A: x, y, z', (0, 0, 0))
        self.line_direction = self.entries(extra, 'Направление: dx, dy, dz', (1, 1, 0))
        self.button(extra, 'Повернуть вокруг прямой', self.rotate_line)

    def collapsible(self, parent, title):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=6)
        expanded = tk.BooleanVar(value=False)
        body = ttk.Frame(section)

        def toggle():
            if expanded.get():
                body.pack(fill='x')
            else:
                body.pack_forget()

        ttk.Checkbutton(section, text=title, variable=expanded, command=toggle).pack(anchor='w')
        return body

    def _camera_controls(self):
        tab = self.camera_tab
        self.position_fields = self.entries(tab, 'Положение x, y, z', (5, 3, 7))
        self.direction_fields = self.entries(tab, 'Направление dx, dy, dz', (-5, -3, -7))
        ttk.Label(tab, text='Проекция').pack(anchor='w', pady=(8, 2))
        combo = ttk.Combobox(tab, textvariable=self.projection, values=['Параллельная', 'Перспективная'], state='readonly')
        combo.pack(fill='x')
        combo.bind('<<ComboboxSelected>>', self.guarded_event(self.set_projection))
        self.button(tab, 'Применить', self.apply_camera)
        self.button(tab, 'Получить текущие значения', self.sync_camera)
        extra = self.collapsible(tab, 'Дополнительно')
        self.target_fields = self.entries(extra, 'Центр орбиты x, y, z', (0, 0, 0))
        self.fov_fields = self.entries(extra, 'FOV, ° / высота кадра', (50, 6))
        self.clip_fields = self.entries(extra, 'Near / far', (.1, 100))
        self.button(extra, 'Матрица проекции', self.show_matrix)
        ttk.Label(tab, textvariable=self.camera_status, wraplength=280).pack(anchor='w', pady=10)

    def show_matrix(self):
        if self.matrix_text is not None and self.matrix_text.winfo_exists():
            self.matrix_text.winfo_toplevel().lift()
            return
        window = tk.Toplevel(self.root)
        window.title('Матрица проекции')
        window.resizable(False, False)
        self.matrix_text = tk.Text(window, width=39, height=4, font=('Consolas', 10), state='disabled')
        self.matrix_text.pack(padx=12, pady=12)
        self.invalidate()

    def select_mode(self):
        self.demo(['Отсечение граней', 'Z-буфер', 'Камера'].index(self.mode.get()) + 1)

    def guarded_event(self, action):
        return lambda _: self.guarded(action)()

    def sync_camera(self):
        for variables, values in [(self.position_fields, self.camera.position),
                                   (self.direction_fields, self.camera.direction),
                                   (self.target_fields, self.target),
                                   (self.fov_fields, [self.camera.fov, self.camera.ortho_height]),
                                   (self.clip_fields, [self.camera.near, self.camera.far])]:
            for variable, value in zip(variables, values):
                variable.set(f'{value:.6g}')
        self.projection.set('Перспективная' if self.camera.projection == 'perspective' else 'Параллельная')

    def set_projection(self):
        self.camera.projection = 'perspective' if self.projection.get() == 'Перспективная' else 'orthographic'
        if self.camera.projection != 'perspective' and self.animation.get() == 'camera':
            self.animation.set('none')
        self.invalidate()

    def apply_camera(self):
        position = np.array(self.values(self.position_fields))
        direction = np.array(self.values(self.direction_fields))
        target = np.array(self.values(self.target_fields))
        fov, height = self.values(self.fov_fields)
        near, far = self.values(self.clip_fields)
        if height <= 0:
            raise ValueError('Высота параллельного кадра должна быть положительной.')
        candidate = Camera(position, direction,
                           'perspective' if self.projection.get() == 'Перспективная' else 'orthographic',
                           fov, height, near, far)
        candidate.view_matrix()
        candidate.projection_matrix(1)
        self.camera, self.target = candidate, target
        if self.animation.get() == 'camera':
            self.animation.set('none')

    def read_light(self):
        try:
            position = np.array(self.values(self.light_pos))
            color = tuple(int(float(v.get())) for v in self.light_color)
            ambient = float(self.ambient.get().replace(',', '.'))
            light = Light(position, color, ambient)
        except (ValueError, tk.TclError, OverflowError):
            return self.light
        self.light = light
        return light

    def current_texture(self):
        if self.texture_name.get() == 'Файл' and self.custom_texture is not None:
            return self.custom_texture
        if self.texture_name.get() == 'Круги':
            return self.circle_texture
        return self.checker_texture

    def apply_object_color(self):
        color = tuple(int(float(v.get())) for v in self.object_color)
        if any(channel < 0 or channel > 255 for channel in color):
            raise ValueError('Каналы цвета объекта должны быть от 0 до 255.')
        self.current().color = color

    def refresh_list(self):
        self.object_list.delete(0, 'end')
        for i, obj in enumerate(self.objects):
            self.object_list.insert('end', f'{i + 1}. {obj.mesh.name}')
        self.selected = max(0, min(self.selected, len(self.objects) - 1))
        if self.objects:
            self.object_list.selection_set(self.selected)
        self.update_model_info()
        self.sync_object_color()
        self.invalidate()

    def update_model_info(self):
        if not self.objects:
            self.model_info.set('Сцена пуста.')
            return
        mesh = self.objects[self.selected].mesh
        self.model_info.set(f'Вершин: {len(mesh.vertices)}   Граней: {len(mesh.faces)}')

    def sync_object_color(self):
        if not self.objects:
            return
        for variable, channel in zip(self.object_color, self.objects[self.selected].color):
            variable.set(str(int(channel)))

    def select_object(self, _event):
        selection = self.object_list.curselection()
        if selection:
            self.selected = selection[0]
            self.update_model_info()
            self.sync_object_color()

    def current(self):
        if not self.objects:
            raise ValueError('Сначала добавьте объект в сцену.')
        return self.objects[self.selected]

    def add_mesh(self, mesh, replace=False, color=None):
        colors = [(75, 162, 227), (237, 141, 65), (98, 193, 144), (179, 122, 216)]
        if replace:
            self.objects = []
        chosen = color if color is not None else colors[len(self.objects) % len(colors)]
        self.objects.append(SceneObject(mesh, chosen))
        self.selected = len(self.objects) - 1
        self.refresh_list()

    def add_model(self, replace=False):
        self.add_mesh(make_mesh(self.model_name.get()), replace)

    def open_obj(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=BASE_DIR,
                                          filetypes=[('Wavefront OBJ', '*.obj')])
        if path:
            mesh = load_obj(path)
            self.add_mesh(mesh.normalized() if self.normalize_obj.get() else mesh)

    def open_texture(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=BASE_DIR,
                                          filetypes=[('Изображения', '*.png *.jpg *.jpeg *.bmp *.webp')])
        if not path:
            return
        image = np.asarray(Image.open(path).convert('RGB'))
        if image.size == 0:
            raise ValueError('Пустое изображение.')
        self.custom_texture = image
        self.texture_name.set('Файл')
        self.shading.set('Текстура')

    def export_obj(self):
        obj = self.current()
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=BASE_DIR,
                    initialfile=obj.mesh.name + '.obj', defaultextension='.obj', filetypes=[('Wavefront OBJ', '*.obj')])
        if path:
            save_obj(obj, path)

    def remove_object(self):
        self.current()
        self.objects.pop(self.selected)
        self.refresh_list()

    def build_revolution(self):
        n = int(self.divisions.get())
        if not 3 <= n <= 96:
            raise ValueError('Количество разбиений: от 3 до 96.')
        profile = PROFILE_PRESETS[self.profile.get()]
        is_torus = self.profile.get() == 'Тор (профиль)'
        if is_torus:
            poly = build_surface_of_revolution(profile[:-1], 'y', n)
            last = len(poly.points) - n
            for j in range(n):
                poly.polygons.append(Polygon([last + j, j, (j + 1) % n, last + (j + 1) % n]))
        else:
            profile = [(0, profile[0][1], 0)] + list(profile) + [(0, profile[-1][1], 0)]
            poly = build_surface_of_revolution(profile, 'y', n)
        poly.name = self.profile.get() + ' (ЛР7)'
        self.add_mesh(Mesh.from_polyhedron(poly), True)

    def build_graph(self):
        name = self.function.get()
        poly = build_graph_surface(SURFACE_FUNCTIONS[name], -math.pi, math.pi,
                                   -math.pi, math.pi, 24, 24, name)
        self.add_mesh(Mesh.from_polyhedron(poly).normalized(), True)
        self.cull.set(False)

    def show_textured_polyhedra(self):
        self.shading.set('Текстура')
        self.texture_name.set('Шахматка')
        self.objects = []
        for name, shift in (('Тетраэдр', -2.7), ('Гексаэдр (куб)', 0.0), ('Октаэдр', 2.7)):
            obj = SceneObject(make_mesh(name), (255, 255, 255))
            obj.apply(Matrix4.translation(shift, 0, 0))
            self.objects.append(obj)
        self.selected = 1
        self.cull.set(True)
        self.zbuffer.set(True)
        self.camera.projection = 'perspective'
        self.projection.set('Перспективная')
        self.animation.set('object')
        self.refresh_list()
        self.fit_scene()

    def transform(self, matrix, about_center=False):
        self.current().apply(matrix, about_center)

    def rotate_selected(self):
        angle, = self.values(self.angle)
        matrix = getattr(Matrix4, 'rotation_' + self.axis.get())(angle)
        self.transform(matrix, True)

    def rotate_line(self):
        point = self.values(self.line_point)
        direction = self.values(self.line_direction)
        angle, = self.values(self.angle)
        self.transform(Matrix4.rotation_around_axis_line(point, direction, angle))

    def reset_object(self):
        self.current().model = np.eye(4)

    def reverse_objects(self):
        selected = self.current()
        self.objects.reverse()
        self.selected = next(i for i, obj in enumerate(self.objects) if obj is selected)
        self.refresh_list()

    def fit_scene(self):
        if not self.objects:
            return
        vertices = np.concatenate([obj.world_vertices() for obj in self.objects])
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        radius = max(float(np.linalg.norm(vertices - center, axis=1).max()), .5)
        self.target = center
        self.camera.position = center - unit(self.camera.direction) * radius * 3.4
        self.camera.direction = center - self.camera.position
        self.camera.ortho_height = radius * 2.6
        self.camera.near = max(.01, radius / 100)
        self.camera.far = max(100, radius * 20)
        self.light.position = center + np.array([0.85, 1.25, 0.65]) * radius
        for variable, value in zip(self.light_pos, self.light.position):
            variable.set(f'{float(value):.4g}')
        self.sync_camera()

    def lighting_demo(self):
        self.camera = Camera()
        self.target = np.zeros(3)
        self.objects = [SceneObject(make_mesh('Сфера (OBJ)'), (222, 170, 120))]
        self.selected = 0
        self.cull.set(True)
        self.zbuffer.set(True)
        self.edges.set(False)
        self.normals.set(False)
        self.depth_view.set(False)
        self.animation.set('object')
        self.shading.set('Гуро (Ламберт)')
        self.mode.set('Z-буфер')
        self.refresh_list()
        self.sync_camera()
        self.notebook.select(self.light_tab)

    def demo(self, number):
        self.camera = Camera(projection='orthographic' if number == 1 else 'perspective')
        self.target = np.zeros(3)
        self.objects = ([SceneObject(make_mesh('Ваза (ЛР7)'), (222, 170, 120))] if number != 2 else demo_scene())
        self.selected = 0
        self.cull.set(True)
        self.zbuffer.set(number != 1)
        self.edges.set(False)
        self.normals.set(False)
        self.depth_view.set(False)
        self.animation.set({1: 'object', 2: 'none', 3: 'camera'}[number])
        self.mode.set({1: 'Отсечение граней', 2: 'Z-буфер', 3: 'Камера'}[number])
        self.refresh_list()
        self.sync_camera()
        self.notebook.select(self.camera_tab if number == 3 else self.scene_tab)
        self.invalidate()

    def animation_changed(self):
        if self.animation.get() == 'camera':
            if np.linalg.norm(self.camera.position - self.target) < 1e-8:
                self.camera.position = self.target + np.array([5., 3., 7.])
            self.camera.projection = 'perspective'
            self.projection.set('Перспективная')
        self.invalidate()

    def toggle_animation(self, event=None):
        if event is not None and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
            return
        self.animation.set('none' if self.animation.get() != 'none' else 'object')
        self.animation_changed()
        return 'break'

    def drag_start(self, event):
        self.mouse = (event.x, event.y)
        self.animation.set('none')

    def drag(self, event):
        if not hasattr(self, 'mouse'):
            return
        dx, dy = event.x - self.mouse[0], event.y - self.mouse[1]
        self.mouse = (event.x, event.y)
        offset = self.camera.position - self.target
        radius = max(float(np.linalg.norm(offset)), .2)
        azimuth = math.atan2(offset[0], offset[2]) - dx * .008
        elevation = np.clip(math.asin(np.clip(offset[1] / radius, -1, 1)) + dy * .008, -1.5, 1.5)
        self.camera.position = self.target + radius * np.array([
            math.cos(elevation) * math.sin(azimuth), math.sin(elevation), math.cos(elevation) * math.cos(azimuth)])
        self.camera.direction = self.target - self.camera.position
        self.invalidate()

    def zoom(self, event, step=None):
        step = step if step is not None else np.sign(event.delta)
        factor = 0.9 ** step
        if self.camera.projection == 'perspective':
            offset = self.camera.position - self.target
            distance = float(np.linalg.norm(offset))
            if distance > 1e-8:
                new_distance = np.clip(distance * factor, self.camera.near * 2, self.camera.far * .8)
                self.camera.position = self.target + offset * new_distance / distance
                self.camera.direction = self.target - self.camera.position
        else:
            self.camera.ortho_height = float(np.clip(self.camera.ortho_height * factor, .05, 10000))
        self.invalidate()

    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last_time, .1)
        self.last_time = now
        try:
            speed = float(self.speed.get().replace(',', '.'))
            if not math.isfinite(speed) or abs(speed) > 360:
                speed = 0
        except ValueError:
            speed = 0
        mode = self.animation.get()
        if mode == 'object' and self.objects and speed:
            self.objects[self.selected].apply(Matrix4.rotation_y(speed * dt), about_center=True)
            self.dirty = True
        elif mode == 'camera' and speed:
            self.camera.orbit(self.target, speed * dt)
            self.dirty = True
        if self.dirty:
            self.render()
        self.root.after(30, self.tick)

    def render(self):
        start = time.perf_counter()
        w, h = max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())
        factor = min(1, 900 / w, 700 / h)
        rw, rh = max(2, int(w * factor)), max(2, int(h * factor))
        renderer = SoftwareRenderer(rw, rh)
        light = self.read_light()
        result = renderer.render(
            self.objects, self.camera, self.cull.get(), self.zbuffer.get(),
            self.edges.get(), self.normals.get(), SHADING_MODES[self.shading.get()],
            light, self.current_texture(), self.light_texture.get())
        self.last_result = result
        pixels = result.depth_image() if self.depth_view.get() else result.color
        image = Image.fromarray(pixels).resize((w, h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        if self.normals.get() and not self.depth_view.get():
            for line, front in result.normal_lines:
                a, b = line * np.array([w / rw, h / rh])
                self.canvas.create_line(*a, *b, fill='#177d4b' if front else '#dc5865', width=2, arrow='last')
        if self.show_light.get() and not self.depth_view.get():
            marker = project_to_pixels(light.position, self.camera, rw, rh)
            if marker is not None:
                x, y = marker[0] * w / rw, marker[1] * h / rh
                self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill='#ffd45a', outline='#8a5a00')
                self.canvas.create_text(x + 12, y - 12, text='свет', fill='#8a5a00', anchor='w')
        self.draw_axes(w, h)
        s = result.stats
        ms = (time.perf_counter() - start) * 1000
        self.status.set(f'{self.shading.get()}   Грани: {s["faces"]}   Отсечено: {s["culled"]}   {ms:.0f} мс')
        p, d = self.camera.position, unit(self.camera.direction)
        self.camera_status.set(f'Положение: {p[0]:.2f}; {p[1]:.2f}; {p[2]:.2f}\n'
                               f'Направление: {d[0]:.2f}; {d[1]:.2f}; {d[2]:.2f}')
        if self.matrix_text is not None and self.matrix_text.winfo_exists():
            matrix = self.camera.projection_matrix(rw / rh)
            self.matrix_text.configure(state='normal')
            self.matrix_text.delete('1.0', 'end')
            self.matrix_text.insert('1.0', '\n'.join(' '.join(f'{v:8.3f}' for v in row) for row in matrix))
            self.matrix_text.configure(state='disabled')
        self.dirty = False

    def draw_axes(self, w, h):
        origin = np.array([55., h - 55.])
        view = self.camera.view_matrix()[:3, :3]
        for i, (label, color) in enumerate([('X', '#c64b5b'), ('Y', '#22825d'), ('Z', '#3b70b4')]):
            endpoint = origin + np.array([view[0, i], -view[1, i]]) * 36
            self.canvas.create_line(*origin, *endpoint, fill=color, width=2, arrow='last')
            self.canvas.create_text(*(endpoint + [5, -6]), text=label, fill=color)

    def save_image(self):
        if self.last_result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=BASE_DIR,
                    initialfile='lab9.png', defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            data = self.last_result.depth_image() if self.depth_view.get() else self.last_result.color
            Image.fromarray(data).save(path)


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
