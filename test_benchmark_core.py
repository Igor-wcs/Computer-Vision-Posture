import csv
import unittest
import tempfile
from pathlib import Path

from benchmark_core import (
    FORWARD,
    GOOD_POSTURE,
    HEAD_DOWN,
    LOG_COLUMNS,
    SHOULDERS,
    TOO_FAR,
    GuidedProtocol,
    ProtocolStep,
    SessionLogger,
    archive_generated_files,
    classify_posture,
)
from reporting import generate_report, missing_logs


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ClassificationTests(unittest.TestCase):
    def test_all_postures(self):
        base_head = 0.20
        base_width = 0.30
        self.assertEqual(classify_posture((0.35, 0.60), (0.65, 0.60), 0.20, 0.30, base_head, base_width), GOOD_POSTURE)
        self.assertEqual(classify_posture((0.35, 0.50), (0.65, 0.60), 0.20, 0.30, base_head, base_width), SHOULDERS)
        self.assertEqual(classify_posture((0.35, 0.60), (0.65, 0.60), 0.10, 0.30, base_head, base_width), HEAD_DOWN)
        self.assertEqual(classify_posture((0.30, 0.60), (0.70, 0.60), 0.20, 0.40, base_head, base_width), FORWARD)
        self.assertEqual(classify_posture((0.40, 0.60), (0.60, 0.60), 0.20, 0.20, base_head, base_width), TOO_FAR)


class ProtocolTests(unittest.TestCase):
    def test_only_advances_after_expected_pose_is_held(self):
        clock = FakeClock()
        protocol = GuidedProtocol((ProtocolStep("neutra", GOOD_POSTURE, 1.0), ProtocolStep("longe", TOO_FAR, 1.0)), time_fn=clock)
        self.assertFalse(protocol.update(GOOD_POSTURE))
        clock.advance(0.8)
        self.assertFalse(protocol.update(GOOD_POSTURE))
        protocol.update(TOO_FAR)
        clock.advance(0.3)
        protocol.update(GOOD_POSTURE)
        clock.advance(1.0)
        self.assertTrue(protocol.update(GOOD_POSTURE))
        self.assertEqual(protocol.index, 1)
        protocol.update(TOO_FAR)
        clock.advance(1.0)
        self.assertTrue(protocol.update(TOO_FAR))
        self.assertTrue(protocol.completed)


class LoggingAndReportTests(unittest.TestCase):
    def _write_log(self, path, multiplier):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            for index in range(4):
                writer.writerow(
                    {
                        "Timestamp": f"2026-01-01T00:00:0{index}-03:00",
                        "FPS": (20 + index) * multiplier,
                        "CPU_Usage_Percent": 10 + index * 50,
                        "RAM_Usage_MB": 200 + index,
                        "Posture_Status": GOOD_POSTURE,
                        "Nose_X": 0.5,
                        "Nose_Y": 0.4 + index * 0.01,
                        "Head_Dist": 0.2,
                    }
                )

    def test_logger_only_writes_while_active(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "log.csv"
            logger = SessionLogger(path, interval_seconds=1.0)
            self.assertFalse(logger.record(fps=1, cpu=1, ram_mb=1, status=GOOD_POSTURE, nose_x=0, nose_y=0, head_distance=0, now=1))
            logger.start()
            self.assertTrue(logger.record(fps=20, cpu=10, ram_mb=200, status=GOOD_POSTURE, nose_x=.5, nose_y=.4, head_distance=.2, now=1))
            self.assertFalse(logger.record(fps=20, cpu=10, ram_mb=200, status=GOOD_POSTURE, nose_x=.5, nose_y=.4, head_distance=.2, now=1.5))
            logger.stop()
            with path.open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_report_and_archive(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            project = Path(temp)
            self._write_log(project / "log_mediapipe.csv", 1.0)
            self._write_log(project / "log_yolo.csv", 0.6)
            self._write_log(project / "log_yolo_onnx.csv", 0.8)
            self._write_log(project / "log_yolo_openvino.csv", 0.9)
            self.assertEqual(missing_logs(project), [])
            result = generate_report(project)
            self.assertTrue(result["image"].exists())
            self.assertTrue(result["summary_csv"].exists())
            self.assertTrue(result["markdown"].exists())
            self.assertIn("FPS e CPU", result["markdown"].read_text(encoding="utf-8"))
            cpu_rows = result["summary"][result["summary"]["Metrica"] == "CPU (%)"]
            self.assertLessEqual(float(cpu_rows["Maximo"].max()), 100.0)
            moved = archive_generated_files(project)
            self.assertEqual(len(moved), 7)
            self.assertTrue(all(path.exists() for path in moved))


if __name__ == "__main__":
    unittest.main()
