import unittest

import numpy as np

from onnx_pose import decode_pose_output, preprocess


class OnnxPoseTests(unittest.TestCase):
    def test_preprocess_preserves_expected_onnx_tensor_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, scale, left, top = preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(scale, 1.0)
        self.assertEqual(left, 0)
        self.assertEqual(top, 80)

    def test_decode_uses_highest_confidence_person_and_normalizes_points(self):
        output = np.zeros((1, 56, 2), dtype=np.float32)
        output[0, :5, 0] = [320, 320, 200, 300, 0.90]
        output[0, 5:8, 0] = [320, 200, 0.95]  # nariz
        output[0, 20:23, 0] = [280, 350, 0.95]  # ombro esquerdo (indice 5)
        output[0, 23:26, 0] = [360, 350, 0.95]  # ombro direito (indice 6)
        output[0, :5, 1] = [100, 100, 10, 10, 0.10]
        detection = decode_pose_output(output, (640, 640), 1.0, 0, 0)
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.score, 0.90, places=4)
        self.assertAlmostEqual(float(detection.keypoints[0, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(detection.keypoints[0, 1]), 0.3125, places=4)
        self.assertAlmostEqual(float(detection.keypoints[5, 0]), 0.4375, places=4)


if __name__ == "__main__":
    unittest.main()
