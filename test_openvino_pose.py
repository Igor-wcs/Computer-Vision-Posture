import tempfile
import unittest
from pathlib import Path

from openvino_pose import check_availability


class OpenVinoAvailabilityTests(unittest.TestCase):
    def test_model_absence_hides_the_backend(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            available, reason, device = check_availability(Path(temp) / "missing.xml")
        self.assertFalse(available)
        self.assertIsNone(device)
        self.assertIn("ausente", reason)


if __name__ == "__main__":
    unittest.main()
