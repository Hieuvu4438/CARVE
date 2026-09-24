"""Fast Phase-1 checks that do not train or evaluate real data."""

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENT / "scripts"))
sys.path.insert(0, str(EXPERIMENT))

import torch
import torch.nn as nn

from clave.verifier import Verifier
from common import EXPERIMENT_ROOT, SOURCE_ROOT, candidate_path, run_dir
from freeze_and_test import REGISTERED, dimensions, finish_test_log, prediction_name, \
    prepare_test_log, proposer_seed


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)

    def resize_token_embeddings(self, size):
        self.embedding = nn.Embedding(size, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def released_verifier():
    spec = importlib.util.spec_from_file_location(
        "released_verifier_for_test", SOURCE / "clave" / "verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Verifier


class Phase1StaticTests(unittest.TestCase):
    def test_read_only_paths(self):
        self.assertEqual(EXPERIMENT_ROOT, EXPERIMENT)
        self.assertEqual(SOURCE_ROOT, SOURCE)
        self.assertEqual(run_dir("verifier_s42"), SOURCE / "runs" / "verifier_s42")
        self.assertEqual(candidate_path("dev_s42.jsonl"),
                         SOURCE / "data" / "cands" / "dev_s42.jsonl")
        self.assertEqual(candidate_path("dev_s13.jsonl", write=True),
                         EXPERIMENT / "data" / "cands" / "dev_s13.jsonl")
        with self.assertRaises(ValueError):
            run_dir("verifier_s42", write=True)
        self.assertEqual(proposer_seed("verifier_s42"), 42)

    def test_registered_decoding_constraints(self):
        args = argparse.Namespace(
            nms_modes=None, fix=["role_alpha=1.0"],
            rule_name="rule_abl_role_verifier.json", runs=REGISTERED["rule_abl_role_verifier.json"])
        grids, fixed = dimensions(args)
        self.assertEqual(grids["role_alpha"], [1.0])
        self.assertEqual(fixed, {"role_alpha": 1.0})
        args.fix = ["role_alpha=0.0"]
        with self.assertRaises(ValueError):
            dimensions(args)
        self.assertEqual(prediction_name("verifier_s42", "decoding_rule.json", None),
                         "test_verifier_s42.jsonl")
        self.assertEqual(prediction_name("verifier_v13_p13", "decoding_rule_3seed.json", None),
                         "test_verifier_v13_p13__decoding_rule_3seed.jsonl")

    def test_test_look_is_logged_once_before_work(self):
        with tempfile.TemporaryDirectory(dir=EXPERIMENT) as work:
            root = Path(work)
            (root / "TEST_LOG.md").write_text("| Time | ID | Runs | Rule | Frozen | Files | Arg-C | Note |\n")
            (root / "assets").mkdir()
            (root / "assets" / "decoding_rule_3seed.json").write_text('{"frozen_at":"old"}')
            args = argparse.Namespace(rule_name="decoding_rule_3seed.json",
                                      runs=REGISTERED["decoding_rule_3seed.json"],
                                      pred_tag=None, resume=False)
            def local_output(*parts):
                path = root.joinpath(*parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                return path
            with patch("freeze_and_test.EXPERIMENT_ROOT", root), \
                    patch("freeze_and_test.output_path", side_effect=local_output), \
                    patch("freeze_and_test.rule_path", side_effect=lambda name: root / "assets" / name):
                log_path, row, paths = prepare_test_log(args, {"frozen_at": "old"})
                self.assertIn("| RUNNING |", log_path.read_text())
                self.assertEqual(len(paths), 3)
                args.resume = True
                resumed = prepare_test_log(args, {"frozen_at": "old"})
                self.assertEqual(resumed[1], row)
                (root / "assets" / "decoding_rule_3seed.json").write_text('{"frozen_at":"changed"}')
                with self.assertRaises(RuntimeError):
                    prepare_test_log(args, {"frozen_at": "changed"})
                fake = [{"arg_c_iou": {"f1": value}} for value in (50.0, 51.0, 52.0)]
                finish_test_log(log_path, row, fake)
                self.assertIn("51.00 ± 1.00", log_path.read_text())

    def test_default_verifier_rng_and_state_layout(self):
        source_class = released_verifier()
        config = SimpleNamespace(hidden_size=4)
        with patch("clave.verifier.AutoConfig.from_pretrained", return_value=config), \
                patch("clave.verifier.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyEncoder()):
            torch.manual_seed(123)
            released = source_class("unused", 16)
            torch.manual_seed(123)
            experiment = Verifier("unused", 16, use_feats=True)
            self.assertEqual(list(released.state_dict()), list(experiment.state_dict()))
            for key, value in released.state_dict().items():
                self.assertTrue(torch.equal(value, experiment.state_dict()[key]), key)
            text_only = Verifier("unused", 16, use_feats=False)
            self.assertFalse(any(k.startswith("feat.") for k in text_only.state_dict()))
            self.assertEqual(text_only.head[0].in_features, 12)
            x = torch.tensor([[1, 2, 3]])
            out = text_only(x, torch.ones_like(x), torch.tensor([1]), torch.tensor([2]),
                            torch.zeros(1, 17))
            self.assertEqual(tuple(out.shape), (1, 10))


if __name__ == "__main__":
    unittest.main()
