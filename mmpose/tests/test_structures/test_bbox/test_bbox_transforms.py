# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.structures.bbox import (bbox_clip_border, bbox_corner2xyxy,
                                    bbox_xyxy2corner, get_pers_warp_matrix,
                                    get_warp_matrix)


class TestBBoxClipBorder(TestCase):

    def test_bbox_clip_border_2D(self):
        bbox = np.array([[10, 20], [60, 80], [-5, 25], [100, 120]])
        shape = (50, 50)  # Example image shape
        clipped_bbox = bbox_clip_border(bbox, shape)

        expected_bbox = np.array([[10, 20], [50, 50], [0, 25], [50, 50]])

        self.assertTrue(np.array_equal(clipped_bbox, expected_bbox))

    def test_bbox_clip_border_4D(self):
        bbox = np.array([
            [[10, 20, 30, 40], [40, 50, 80, 90]],
            [[-5, 0, 30, 40], [70, 80, 120, 130]],
        ])
        shape = (50, 60)  # Example image shape
        clipped_bbox = bbox_clip_border(bbox, shape)

        expected_bbox = np.array([
            [[10, 20, 30, 40], [40, 50, 50, 60]],
            [[0, 0, 30, 40], [50, 60, 50, 60]],
        ])

        self.assertTrue(np.array_equal(clipped_bbox, expected_bbox))


class TestBBoxXYXY2Corner(TestCase):

    def test_bbox_xyxy2corner_single(self):
        bbox = np.array([0, 0, 100, 50])
        corners = bbox_xyxy2corner(bbox)

        expected_corners = np.array([[0, 0], [0, 50], [100, 0], [100, 50]])

        self.assertTrue(np.array_equal(corners, expected_corners))

    def test_bbox_xyxy2corner_multiple(self):
        bboxes = np.array([[0, 0, 100, 50], [10, 20, 200, 150]])
        corners = bbox_xyxy2corner(bboxes)

        expected_corners = np.array([[[0, 0], [0, 50], [100, 0], [100, 50]],
                                     [[10, 20], [10, 150], [200, 20],
                                      [200, 150]]])

        self.assertTrue(np.array_equal(corners, expected_corners))


class TestBBoxCorner2XYXY(TestCase):

    def test_bbox_corner2xyxy_single(self):

        corners = np.array([[0, 0], [0, 50], [100, 0], [100, 50]])
        xyxy = bbox_corner2xyxy(corners)
        expected_xyxy = np.array([0, 0, 100, 50])

        self.assertTrue(np.array_equal(xyxy, expected_xyxy))

    def test_bbox_corner2xyxy_multiple(self):

        corners = np.array([[[0, 0], [0, 50], [100, 0], [100, 50]],
                            [[10, 20], [10, 150], [200, 20], [200, 150]]])
        xyxy = bbox_corner2xyxy(corners)
        expected_xyxy = np.array([[0, 0, 100, 50], [10, 20, 200, 150]])

        self.assertTrue(np.array_equal(xyxy, expected_xyxy))


class TestGetPersWarpMatrix(TestCase):

    def test_get_pers_warp_matrix_identity(self):
        center = np.array([0, 0])
        translate = np.array([0, 0])
        scale = 1.0
        rot = 0.0
        shear = np.array([0.0, 0.0])
        warp_matrix = get_pers_warp_matrix(center, translate, scale, rot,
                                           shear)

        expected_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   dtype=np.float32)

        self.assertTrue(np.array_equal(warp_matrix, expected_matrix))

    def test_get_pers_warp_matrix_translation(self):
        center = np.array([0, 0])
        translate = np.array([10, 20])
        scale = 1.0
        rot = 0.0
        shear = np.array([0.0, 0.0])
        warp_matrix = get_pers_warp_matrix(center, translate, scale, rot,
                                           shear)

        expected_matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]],
                                   dtype=np.float32)

        self.assertTrue(np.array_equal(warp_matrix, expected_matrix))

    def test_get_pers_warp_matrix_scale_rotation_shear(self):
        center = np.array([0, 0])
        translate = np.array([0, 0])
        scale = 1.5
        rot = 45.0
        shear = np.array([15.0, 30.0])
        warp_matrix = get_pers_warp_matrix(center, translate, scale, rot,
                                           shear)

        expected_matrix = np.array([
            [1.3448632, -0.77645713, 0.],
            [1.6730325, 0.44828773, 0.],
            [0., 0., 1.],
        ],
                                   dtype=np.float32)

        # Use np.allclose to compare floating-point arrays within a tolerance
        self.assertTrue(
            np.allclose(warp_matrix, expected_matrix, rtol=1e-3, atol=1e-3))


class TestGetWarpMatrix(TestCase):

    def test_basic_transformation(self):
        # Test with basic parameters
        center = np.array([100, 100])
        scale = np.array([50, 50])
        rot = 0
        output_size = (200, 200)
        warp_matrix = get_warp_matrix(center, scale, rot, output_size)
        expected_matrix = np.array([[4, 0, -300], [0, 4, -300]])
        np.testing.assert_array_almost_equal(warp_matrix, expected_matrix)

    def test_rotation(self):
        # Test with rotation
        center = np.array([100, 100])
        scale = np.array([50, 50])
        rot = 45  # 45 degree rotation
        output_size = (200, 200)
        warp_matrix = get_warp_matrix(center, scale, rot, output_size)
        expected_matrix = np.array([[2.828427, 2.828427, -465.685303],
                                    [-2.828427, 2.828427, 100.]])
        np.testing.assert_array_almost_equal(warp_matrix, expected_matrix)

    def test_shift(self):
        # Test with shift
        center = np.array([100, 100])
        scale = np.array([50, 50])
        rot = 0
        output_size = (200, 200)
        shift = (0.1, 0.1)  # 10% shift
        warp_matrix = get_warp_matrix(
            center, scale, rot, output_size, shift=shift)
        expected_matrix = np.array([[4, 0, -320], [0, 4, -320]])
        np.testing.assert_array_almost_equal(warp_matrix, expected_matrix)

    def test_inverse(self):
        # Test inverse transformation
        center = np.array([100, 100])
        scale = np.array([50, 50])
        rot = 0
        output_size = (200, 200)
        warp_matrix = get_warp_matrix(
            center, scale, rot, output_size, inv=True)
        expected_matrix = np.array([[0.25, 0, 75], [0, 0.25, 75]])
        np.testing.assert_array_almost_equal(warp_matrix, expected_matrix)

    def test_aspect_ratio(self):
        # Test with fix_aspect_ratio set to False
        center = np.array([100, 100])
        scale = np.array([50, 20])
        rot = 0
        output_size = (200, 200)
        warp_matrix = get_warp_matrix(
            center, scale, rot, output_size, fix_aspect_ratio=False)
        expected_matrix = np.array([[4, 0, -300], [0, 10, -900]])
        np.testing.assert_array_almost_equal(warp_matrix, expected_matrix)
