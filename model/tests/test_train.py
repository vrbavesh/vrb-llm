import unittest

import numpy as np
import torch

from model.config import TRAIN
from model.train import batch_at, lr_at


class ScheduleTests(unittest.TestCase):
    def test_warmup_is_linear_from_zero(self):
        peak, warmup, total = 1e-3, 100, 1000
        self.assertAlmostEqual(lr_at(0, total, peak, warmup, 0.1), peak / warmup)
        self.assertAlmostEqual(lr_at(warmup - 1, total, peak, warmup, 0.1), peak)
        # Step counter is 1-based inside warmup, so step 50 sits at 51/100 of peak.
        self.assertAlmostEqual(lr_at(50, total, peak, warmup, 0.1), peak * 51 / 100)

    def test_decay_is_monotonic_and_ends_at_floor(self):
        peak, warmup, total = 1e-3, 100, 1000
        rates = [lr_at(step, total, peak, warmup, 0.1) for step in range(warmup, total)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        # The cosine asymptote is the floor; the last executed step sits just above it.
        self.assertLessEqual(rates[-1], peak * 0.1001)
        self.assertGreaterEqual(rates[-1], peak * 0.1)

    def test_never_exceeds_peak(self):
        peak = 1e-3
        for step in range(0, 500, 7):
            self.assertLessEqual(lr_at(step, 500, peak, 50, 0.1), peak + 1e-12)


class BatchTests(unittest.TestCase):
    def test_targets_are_inputs_shifted_by_one(self):
        data = np.arange(0, 5000, dtype=np.uint16)
        x, y = batch_at(data, 10, micro=2, seq=8, device="cpu")
        self.assertEqual(x.shape, (2, 8))
        self.assertEqual(y.shape, (2, 8))
        # The flattened stream must be contiguous and targets offset by exactly one.
        flat = x.reshape(-1).tolist()
        self.assertEqual(flat, list(range(10, 26)))
        self.assertEqual(y.reshape(-1).tolist(), list(range(11, 27)))

    def test_batch_keeps_row_boundaries(self):
        data = np.arange(0, 5000, dtype=np.uint16)
        x, _ = batch_at(data, 0, micro=4, seq=3, device="cpu")
        self.assertEqual(x[0].tolist(), [0, 1, 2])
        self.assertEqual(x[1].tolist(), [3, 4, 5])
        self.assertEqual(x[3].tolist(), [9, 10, 11])

    def test_offset_scales_with_step(self):
        # A step covers tokens_per_step tokens, so consecutive steps never overlap.
        self.assertEqual(TRAIN.tokens_per_step, TRAIN.micro_batch * TRAIN.seq_len * TRAIN.grad_accum)
        consumed = TRAIN.total_steps * TRAIN.tokens_per_step
        self.assertLessEqual(consumed, TRAIN.total_tokens)
        self.assertLess(TRAIN.total_tokens - consumed, TRAIN.tokens_per_step)


if __name__ == "__main__":
    unittest.main()
