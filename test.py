from utils import YoloUtils
import unittest


class TestYoloUtils(unittest.TestCase):
    def setUp(self):
        self.utils = YoloUtils(480, 7)

    def test_get_grid_point(self):
        grid_point = self.utils.get_grid_point(0.4, 0.5)
        self.assertEqual(grid_point, (2, 3))
        grid_point = self.utils.get_grid_point(1, 1)
        self.assertEqual(grid_point, (6, 6))
        grid_point = self.utils.get_grid_point(0, 0)
        self.assertEqual(grid_point, (0, 0))

unittest.main()
