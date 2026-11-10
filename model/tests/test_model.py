import unittest
from dataclasses import replace

import torch

from model.config import ARCH, TRAIN
from model.gpt import GPT, GPTConfig, apply_rope, rope_cache


def expected_params(cfg):
    """Vocabulary table plus, per block, attention, SwiGLU, and two RMSNorms."""
    attn = 4 * cfg.d_model**2
    mlp = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    return cfg.vocab_size * cfg.d_model + cfg.n_layer * (attn + mlp + norms) + cfg.d_model


class ConfigTests(unittest.TestCase):
    def test_parameter_budget(self):
        total = GPT(ARCH).num_params()
        self.assertGreaterEqual(total, 45_000_000)
        self.assertLessEqual(total, 50_000_000)

    def test_module_matches_intended_architecture(self):
        self.assertEqual(GPT(ARCH).num_params(), expected_params(ARCH))

    def test_embeddings_are_tied(self):
        model = GPT(ARCH)
        self.assertIs(model.head.weight, model.tok_emb.weight)

    def test_swiglu_width_matches_ratio(self):
        self.assertEqual(ARCH.d_ff % 128, 0)
        target = 8 * ARCH.d_model / 3
        self.assertLess(abs(ARCH.d_ff - target) / target, 0.05)

    def test_config_shape_is_consistent(self):
        self.assertEqual(ARCH.d_model % ARCH.n_head, 0)
        self.assertEqual(ARCH.max_seq_len, TRAIN.seq_len)

    def test_training_schedule_is_integer(self):
        self.assertGreater(TRAIN.total_steps, 0)
        self.assertEqual(TRAIN.total_steps, TRAIN.total_tokens // TRAIN.tokens_per_step)
        self.assertLess(TRAIN.warmup_steps, TRAIN.total_steps // 10)

    def test_unknown_mlp_is_rejected(self):
        with self.assertRaises(ValueError):
            GPTConfig(mlp="relu")

    def test_odd_head_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            GPTConfig(d_model=6, n_head=2)

    def test_default_swiglu_width_is_rounded(self):
        default = GPTConfig().d_ff
        self.assertEqual(default % 128, 0)
        self.assertEqual(default, 1408)


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = GPTConfig(vocab_size=256, max_seq_len=32, n_layer=2, n_head=2, d_model=16)
        torch.manual_seed(0)
        self.model = GPT(self.cfg).eval()

    def test_rope_preserves_norm(self):
        head_dim, seq = 8, 16
        cos, sin = rope_cache(seq, head_dim)
        x = torch.randn(2, 3, seq, head_dim)
        out = apply_rope(x, cos, sin)
        self.assertEqual(out.shape, x.shape)
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)

    def test_rope_at_position_zero_is_identity(self):
        cos, sin = rope_cache(4, 8)
        x = torch.randn(1, 1, 4, 8)
        torch.testing.assert_close(apply_rope(x, cos, sin)[:, :, 0], x[:, :, 0])

    def test_forward_and_loss(self):
        idx = torch.randint(0, self.cfg.vocab_size, (3, 32))
        logits, loss = self.model(idx, idx)
        self.assertEqual(logits.shape, (3, 32, self.cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        # Random init should land near the uniform-guess loss of ln(vocab_size).
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(256.0)).item(), delta=0.5)

    def test_causal_mask_hides_future_tokens(self):
        idx = torch.randint(0, self.cfg.vocab_size, (1, 16))
        base, _ = self.model(idx)
        changed = idx.clone()
        changed[0, -1] = (changed[0, -1] + 1) % self.cfg.vocab_size
        edited, _ = self.model(changed)
        torch.testing.assert_close(base[0, :-1], edited[0, :-1])
        self.assertFalse(torch.allclose(base[0, -1], edited[0, -1]))

    def test_sequence_longer_than_context_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, self.cfg.max_seq_len + 1, dtype=torch.long))

    def test_generation_extends_sequence(self):
        prompt = torch.randint(0, self.cfg.vocab_size, (1, 4))
        out = self.model.generate(prompt, max_new_tokens=5, top_k=10)
        self.assertEqual(out.shape, (1, 9))

    def test_checkpointing_preserves_the_forward_pass(self):
        plain = GPT(self.cfg).train()
        ckpt = GPT(replace(self.cfg, checkpointing=True)).train()
        ckpt.load_state_dict(plain.state_dict())
        idx = torch.randint(0, self.cfg.vocab_size, (2, 8))
        torch.testing.assert_close(plain(idx)[0], ckpt(idx)[0])


if __name__ == "__main__":
    unittest.main()
