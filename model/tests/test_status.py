import os
import tempfile
import time
import unittest
from pathlib import Path

from model.status import STEP_RE, format_duration, read_steps, read_vals, report

STEP_LINE = "step  3048  loss 4.2674  lr 7.77e-05   20892 tok/s  2672 MiB  3739s"


class LogParsingTests(unittest.TestCase):
    def test_step_line_fields(self):
        match = STEP_RE.match(STEP_LINE)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 3048)
        self.assertAlmostEqual(float(match.group(2)), 4.2674)
        self.assertAlmostEqual(float(match.group(3)), 7.77e-05)

    def test_thousands_separator_in_token_rate(self):
        line = "step  100  loss 5.0  lr 1.00e-04   21,056 tok/s  2672 MiB  1200s"
        match = STEP_RE.match(line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(4).replace(",", ""), "21056")

    def test_validation_lines_skip_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                '{"step": 250, "val_loss": 4.0}\n'
                '{"step": 500, "loss": 3.9}\n'
                "not json at all\n"
                '{"step": 750, "val_loss": 3.8}\n',
                encoding="utf-8",
            )
            self.assertEqual(read_vals(path), [(250, 4.0), (750, 3.8)])

    def test_read_steps_ignores_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.log"
            path.write_text(
                "some banner\n  step 250: val_loss 4.0000\n" + STEP_LINE + "\n",
                encoding="utf-8",
            )
            rows = read_steps(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 3048)


class DurationTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(90), "1m 30s")
        self.assertEqual(format_duration(3660), "1h 01m")


class VerdictTests(unittest.TestCase):
    def _run_dir(self, tmp: str, stale_seconds: int) -> Path:
        run_dir = Path(tmp)
        (run_dir / "train.log").write_text(STEP_LINE + "\n", encoding="utf-8")
        when = time.time() - stale_seconds
        os.utime(run_dir / "train.log", (when, when))
        return run_dir

    def test_fresh_log_is_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("verdict  RUNNING", report(self._run_dir(tmp, 5)))

    def test_stale_log_with_traceback_is_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, 3600)
            (run_dir / "train.err").write_text(
                "Traceback (most recent call last):\n"
                "RuntimeError: ios_base::badbit set: iostream stream error\n",
                encoding="utf-8",
            )
            text = report(run_dir)
            self.assertIn("verdict  CRASHED", text)
            self.assertIn("ios_base::badbit", text)
            self.assertIn("--resume", text)

    def test_stale_log_without_traceback_is_stalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("verdict  STALLED", report(self._run_dir(tmp, 3600)))

    def test_partial_write_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, 5)
            (run_dir / "last.pt.tmp").write_bytes(b"x")
            self.assertIn("partial write left behind", report(run_dir))


if __name__ == "__main__":
    unittest.main()
