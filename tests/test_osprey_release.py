"""Guards on the Osprey-specific pieces of the pipeline.

These are the invariants that silently break a run rather than crashing it:
the zero-initialized QKV expansion, the layer count kept per backbone, the
per-target interface presets, and the hyperparameters the paper reports.
"""

import argparse
import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    """Import a `scripts/` entrypoint that is not part of the package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "requires the project's PyTorch dependency")
class OspreyConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen = load_script(
            "osprey_qwen_converter",
            "scripts/convert_qwen3_to_eagle3_preserve_layers.py",
        )
        cls.llama = load_script(
            "osprey_llama_converter",
            "scripts/convert_llama_to_eagle3_preserve_layers.py",
        )

    def test_qkv_expansion_preserves_pretrained_half(self):
        """Eq. 1: [0, W]. Pretrained weights on the hidden-state half, zeros on
        the new target-feature columns, so the forward at init is unchanged."""
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for converter in (self.qwen, self.llama):
            expanded = converter.expand_qkv(weight, hidden_size=4)
            self.assertEqual(expanded.shape, (3, 8))
            torch.testing.assert_close(expanded[:, 4:], weight)
            self.assertEqual(torch.count_nonzero(expanded[:, :4]).item(), 0)

    def test_fc_is_identity_on_last_aux_block(self):
        """fc starts as the identity on the deepest tapped aux block.

        Function preservation comes from the zero target-feature half of Q/K/V,
        not from fc. fc must be non-zero so that half receives gradient: with an
        all-zero fc, RMSNorm(0) = 0 and both dL/dW_qkv[:, :h] and dL/dfc vanish
        identically -- the target-feature path would never train.
        """
        config = {
            "hidden_size": 8,
            "target_hidden_size": 12,
            "torch_dtype": "float32",
        }
        for converter in (self.qwen, self.llama):
            fc = converter.make_fc_weight(config)
            self.assertEqual(tuple(fc.shape), (8, 36))
            shared = min(8, 12)
            block = fc[:shared, 2 * 12 : 2 * 12 + shared]
            self.assertTrue(torch.equal(block, torch.eye(shared)))
            self.assertEqual(int((fc != 0).sum()), shared)

    def test_converted_config_requests_swap_mode(self):
        converted = self.qwen.make_eagle_config(
            {"hidden_size": 4, "num_hidden_layers": 2, "vocab_size": 8},
            draft_vocab_size=8,
            num_draft_layers=2,
            aux_layer_ids=[1, 30, 58],
            target_hidden_size=16,
        )
        self.assertTrue(converted["swap_h_and_emb"])
        self.assertEqual(converted["architectures"], ["Qwen3ForCausalLMEagle3"])
        self.assertEqual(converted["num_hidden_layers"], 2)
        self.assertEqual(converted["target_hidden_size"], 16)
        self.assertEqual(
            converted["eagle_config"]["eagle_aux_hidden_state_layer_ids"],
            [1, 30, 58],
        )


class OspreyEntrypointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.convert = load_script(
            "osprey_convert_entrypoint", "scripts/osprey/convert_checkpoint.py"
        )
        cls.adapt = load_script(
            "osprey_adaptation_entrypoint", "scripts/osprey/run_adaptation.py"
        )

    def _convert_args(self, **overrides):
        args = argparse.Namespace(
            architecture="qwen3",
            source_checkpoint="source",
            output_dir="output",
            target=None,
            target_hidden_size=None,
            target_tokenizer=None,
            aux_layer_ids=None,
            num_layers=None,
            draft_vocab_size=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_conversion_keeps_two_layers_for_the_qwen_backbone(self):
        commands = self.convert.build_commands(
            self.convert.resolve(self._convert_args(target="qwen3-8b"))
        )
        convert_command = commands[0]
        self.assertIn("convert_qwen3_to_eagle3_preserve_layers.py", convert_command[1])
        self.assertEqual(
            convert_command[convert_command.index("--num-draft-layers") + 1], "2"
        )
        self.assertEqual(
            convert_command[convert_command.index("--target-hidden-size") + 1], "4096"
        )
        # Tokenizer alignment runs as a second step against the target's vocab.
        self.assertIn("align_tokenizer.py", commands[1][1])
        self.assertEqual(
            commands[1][commands[1].index("--target-tokenizer") + 1], "Qwen/Qwen3-8B"
        )

    def test_conversion_keeps_four_layers_for_the_layerskip_backbone(self):
        command = self.convert.build_commands(
            self.convert.resolve(
                self._convert_args(architecture="layerskip", target="qwen3-8b")
            )
        )[0]
        self.assertIn("convert_llama_to_eagle3_preserve_layers.py", command[1])
        self.assertEqual(command[command.index("--num-draft-layers") + 1], "4")

    def test_minimax_preset_pins_the_target_interface(self):
        """MiniMax-M2.5 is 62 layers wide at hidden size 3072 and needs its own
        tapped layer ids; getting these wrong trains against the wrong signal."""
        command = self.convert.build_commands(
            self.convert.resolve(self._convert_args(target="minimax-m25"))
        )[0]
        self.assertEqual(command[command.index("--target-hidden-size") + 1], "3072")
        self.assertEqual(command[command.index("--aux-layer-ids") + 1], "1,30,58")

    def _adapt_args(self, **overrides):
        args = argparse.Namespace(
            target="qwen3-8b",
            draft_checkpoint="draft",
            train_data="train.jsonl",
            eval_data=[],
            output_dir="output",
            cache_dir="cache",
            num_gpus=None,
            tp_size=None,
            learning_rate=1e-4,
            num_epochs=3,
            ttt_length=5,
            max_length=4096,
            eval_interval=1000,
            save_interval=5000,
            sglang_attention_backend=None,
            from_scratch=False,
            report_to="none",
            wandb_project=None,
            wandb_name=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_adaptation_uses_the_paper_configuration(self):
        command = self.adapt.build_command(self._adapt_args())
        pairs = {
            "--nproc_per_node": "2",
            "--target-model-path": "Qwen/Qwen3-8B",
            "--num-epochs": "3",
            "--learning-rate": "0.0001",
            "--ttt-length": "5",
            "--max-length": "4096",
            "--warmup-ratio": "0.04",
            "--chat-template": "qwen3-thinking",
            "--attention-backend": "flex_attention",
            "--tp-size": "1",
        }
        for flag, value in pairs.items():
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        # Osprey owns its tokenizer-aligned embedding and keeps training it.
        self.assertIn("--train-with-embeddings", command)
        self.assertEqual(command[command.index("--ckpt-dir") + 1], "draft")

    def test_from_scratch_baseline_borrows_and_freezes_the_target_embedding(self):
        command = self.adapt.build_command(
            self._adapt_args(from_scratch=True, draft_checkpoint="configs/x.json")
        )
        self.assertNotIn("--train-with-embeddings", command)
        self.assertNotIn("--ckpt-dir", command)
        self.assertEqual(
            command[command.index("--embedding-key") + 1], "model.embed_tokens.weight"
        )
        # A from-scratch arm is built from a plain config file, not a checkpoint.
        self.assertEqual(
            command[command.index("--draft-model-config") + 1], "configs/x.json"
        )

    def test_minimax_adaptation_uses_expert_parallelism(self):
        command = self.adapt.build_command(self._adapt_args(target="minimax-m25"))
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "4")
        self.assertEqual(command[command.index("--tp-size") + 1], "4")
        self.assertEqual(command[command.index("--sglang-ep-size") + 1], "4")
        self.assertEqual(command[command.index("--chat-template") + 1], "minimax-m25")
        self.assertIn("--trust-remote-code", command)


if __name__ == "__main__":
    unittest.main()
