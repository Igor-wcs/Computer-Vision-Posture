"""Testes dos limites usados pelo backend YOLO, sem carregar o modelo."""

import unittest

from benchmark_core import FORWARD, GOOD_POSTURE, HEAD_DOWN, SHOULDERS, classify_posture


class TestPostureYOLOLogic(unittest.TestCase):
    def test_good_posture_logic(self):
        self.assertEqual(classify_posture((0.35, 0.60), (0.65, 0.60), 0.20, 0.30, 0.20, 0.30), GOOD_POSTURE)

    def test_head_down_logic(self):
        self.assertEqual(classify_posture((0.35, 0.60), (0.65, 0.60), 0.05, 0.30, 0.20, 0.30), HEAD_DOWN)

    def test_shoulder_misalignment_logic(self):
        self.assertEqual(classify_posture((0.35, 0.50), (0.65, 0.60), 0.20, 0.30, 0.20, 0.30), SHOULDERS)

    def test_forward_lean_logic(self):
        self.assertEqual(classify_posture((0.30, 0.60), (0.70, 0.60), 0.20, 0.40, 0.20, 0.30), FORWARD)


if __name__ == "__main__":
    unittest.main()
