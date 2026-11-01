"""Testes da classificacao compartilhada, sem abrir GUI ou webcam."""

import unittest

from benchmark_core import GOOD_POSTURE, HEAD_DOWN, SHOULDERS, classify_posture


class TestPostureLogic(unittest.TestCase):
    def test_good_posture_logic(self):
        status = classify_posture((0.35, 0.60), (0.65, 0.60), 0.20, 0.30, 0.20, 0.30)
        self.assertEqual(status, GOOD_POSTURE)

    def test_head_down_logic(self):
        status = classify_posture((0.35, 0.60), (0.65, 0.60), 0.05, 0.30, 0.20, 0.30)
        self.assertEqual(status, HEAD_DOWN)

    def test_shoulder_misalignment_logic(self):
        status = classify_posture((0.35, 0.50), (0.65, 0.60), 0.20, 0.30, 0.20, 0.30)
        self.assertEqual(status, SHOULDERS)


if __name__ == "__main__":
    unittest.main()
