import unittest

import numpy as np

from graphics3d import (BASE_DIR, Camera, Light, Matrix4, Mesh, SceneObject, SoftwareRenderer,
                        bilinear_interpolate, compute_vertex_normals, interpolated_normal,
                        lambert_intensity, load_obj, make_mesh, sample_texture, toon_intensity)


def colored_pixels(image):
    background = np.array([239, 243, 248], dtype=np.uint8)
    flat = image.reshape(-1, 3)
    mask = np.any(flat != background, axis=1)
    return flat[mask]


class NormalTests(unittest.TestCase):
    def test_computed_vertex_normals_point_outward(self):
        mesh = make_mesh('Тетраэдр')
        center = mesh.vertices.mean(axis=0)
        direct = compute_vertex_normals(mesh.vertices, mesh.faces, mesh.normals)
        for face, corners in zip(mesh.faces, mesh.corner_normals):
            for vertex, normal in zip(face, corners):
                self.assertAlmostEqual(np.linalg.norm(normal), 1, places=5)
                self.assertGreater(np.dot(mesh.vertices[vertex] - center, normal), 0.2)
                np.testing.assert_allclose(normal, direct[vertex], atol=1e-6)

    def test_obj_cube_keeps_face_normals_instead_of_averaging(self):
        mesh = load_obj(BASE_DIR / 'cube.obj')
        found = {tuple(np.round(normal, 4)) for face in mesh.corner_normals for normal in face}
        for axis in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            self.assertIn(axis, found)

    def test_obj_sphere_uses_smooth_normals(self):
        mesh = make_mesh('Сфера (OBJ)')
        center = mesh.vertices.mean(axis=0)
        for face, corners in zip(mesh.faces, mesh.corner_normals):
            for vertex, normal in zip(face, corners):
                self.assertAlmostEqual(np.linalg.norm(normal), 1, places=4)
                self.assertGreater(np.dot(normal, mesh.vertices[vertex] - center), 0.9)


class ShadingTests(unittest.TestCase):
    def setUp(self):
        self.renderer = SoftwareRenderer(180, 140)
        self.light = Light(np.array([0.0, 0.0, 6.0]), (255, 255, 255), 0.0)
        self.camera = Camera(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]),
                             'orthographic', ortho_height=4)

    def test_bilinear_weights_and_normal_renormalization(self):
        values = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
        mixed = bilinear_interpolate(values, np.array([0.2]), np.array([0.5]), np.array([0.3]))
        np.testing.assert_allclose(mixed, [[2.0, 5.0]])
        normals = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        normal = interpolated_normal(normals, 0.2, 0.3, 0.5)
        self.assertAlmostEqual(np.linalg.norm(normal), 1, places=6)
        raw = 0.2 * normals[0] + 0.3 * normals[1] + 0.5 * normals[2]
        self.assertGreater(abs(np.linalg.norm(raw) - 1), 0.2)

    def test_gouraud_interpolates_vertex_colors(self):
        image = np.zeros((140, 180, 3), dtype=np.uint8)
        depth = np.full((140, 180), np.inf)
        self.renderer.rasterize_triangle(
            [[0, 0, .2], [100, 0, .5], [0, 100, .5]], None, image, depth,
            attributes=[[255, 0, 0], [0, 255, 0], [0, 0, 255]], shader=lambda samples: samples)
        pixel = image[10, 10].astype(int)
        self.assertGreater(pixel[0], 180)
        self.assertGreater(pixel[1], 10)
        self.assertGreater(pixel[0], pixel[1])
        self.assertGreater(pixel[0], pixel[2])

    def test_lambert_and_toon_are_different_models(self):
        self.assertAlmostEqual(lambert_intensity(0.6, 0.0), 0.6)
        self.assertIn(toon_intensity(0.6, 0.0), (0.35, 0.68, 1.0))
        self.assertNotAlmostEqual(toon_intensity(0.6, 0.0), 0.6)

    def test_reflection_changes_lambert_brightness(self):
        mesh = Mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2, 3]], 'Квадрат')
        obj = SceneObject(mesh, (255, 255, 255))
        before = self.renderer.render([obj], self.camera, shading='gouraud', light=self.light)
        self.assertGreater(colored_pixels(before.color)[:, 0].mean(), 200)
        obj.apply(Matrix4.reflection('xy'))
        after = self.renderer.render([obj], self.camera, cull=False, shading='gouraud', light=self.light)
        self.assertLess(colored_pixels(after.color)[:, 0].mean(), 5)

    def test_back_faces_are_removed_in_every_shading_mode(self):
        camera = Camera(np.array([0., 0., 5.]), np.array([0., 0., -1.]), 'orthographic')
        obj = SceneObject(make_mesh('Куб (OBJ)'))
        for shading in ('gouraud', 'phong', 'texture', 'flat'):
            result = self.renderer.render([obj], camera, shading=shading, light=self.light)
            self.assertEqual(result.stats['culled'], 5, shading)
            self.assertGreater(result.stats['pixels'], 0)

    def test_phong_toon_uses_few_colors_and_gouraud_many(self):
        camera = Camera()
        light = Light(np.array([3.5, 4.5, 5.0]), ambient=0.12)
        obj = SceneObject(make_mesh('Сфера (OBJ)'), (220, 170, 120))
        phong = self.renderer.render([obj], camera, shading='phong', light=light)
        gouraud = self.renderer.render([obj], camera, shading='gouraud', light=light)
        self.assertLess(len(np.unique(colored_pixels(phong.color), axis=0)), 16)
        self.assertGreater(len(np.unique(colored_pixels(gouraud.color), axis=0)), 40)

    def test_texture_coordinates_pick_left_and_right_colors(self):
        texture = np.zeros((16, 16, 3), dtype=np.uint8)
        texture[:, :8] = (220, 20, 20)
        texture[:, 8:] = (20, 20, 220)
        mesh = Mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2, 3]], 'Квадрат')
        light = Light(np.array([0.0, 0.0, 5.0]), ambient=1.0)
        image = self.renderer.render(
            [SceneObject(mesh, (255, 255, 255))], self.camera,
            shading='texture', light=light, texture=texture, light_texture=False).color
        pixels = colored_pixels(image)
        self.assertTrue(np.any(np.all(np.abs(pixels.astype(int) - (220, 20, 20)) < 8, axis=1)))
        self.assertTrue(np.any(np.all(np.abs(pixels.astype(int) - (20, 20, 220)) < 8, axis=1)))
        sampled = sample_texture(texture, [[0.1, 0.5], [0.9, 0.5]])
        self.assertGreater(sampled[0, 0], 200)
        self.assertGreater(sampled[1, 2], 200)

    def test_affine_rotation_changes_textured_picture(self):
        mesh = make_mesh('Гексаэдр (куб)')
        light = Light(np.array([2.0, 3.0, 4.0]), ambient=0.2)
        camera = Camera()
        obj = SceneObject(mesh, (255, 255, 255))
        before = self.renderer.render([obj], camera, shading='texture', light=light)
        obj.apply(Matrix4.rotation_y(50), about_center=True)
        after = self.renderer.render([obj], camera, shading='texture', light=light)
        self.assertGreater(np.count_nonzero(before.color != after.color), 100)


if __name__ == '__main__':
    unittest.main()
