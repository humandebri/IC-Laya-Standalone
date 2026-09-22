import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from quantize_pack import quantize
from canister_infer import uleb, decode_blobs
from laya_port_bridge import translate_config


class Int8PackTests(unittest.TestCase):
    def test_fixture_is_reproducible_and_row_error_is_bounded(self):
        for suffix in ["prenorm", "postnorm"]:
            source = ROOT / "fixtures" / f"tiny-{suffix}"
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "pack"
                result = quantize(source, out)
                expected = ROOT / "fixtures" / f"tiny-int8-{suffix}"
                for name in ["model.bin", "manifest.json", "tokenizer.json"]:
                    self.assertEqual((out / name).read_bytes(), (expected / name).read_bytes())
                old = json.loads((source / "manifest.json").read_text())
                self.assertLess(result["total_bytes"], old["total_bytes"] / 2)
                original = (source / "model.bin").read_bytes()
                packed = (out / "model.bin").read_bytes()
                for a, b in zip(old["tensors"], result["tensors"]):
                    raw = packed[b["offset"]:b["offset"]+b["length"]]
                    self.assertEqual(list(hashlib.sha256(raw).digest()), b["sha256"])
                    if b["storage"] == "I8Row":
                        shape = b["shape"]
                        n = shape[0] * shape[1]
                        q = np.frombuffer(raw[:n], dtype=np.int8).reshape(shape)
                        scale = np.frombuffer(raw[n:], dtype="<f4")
                        f = np.frombuffer(original[a["offset"]:a["offset"]+a["length"]], dtype="<f4").reshape(shape)
                        self.assertTrue(np.all(np.abs(q*scale[:, None] - f) <= scale[:, None]*0.501))
                with self.assertRaises(ValueError):
                    quantize(source, out)

    def test_corrupted_source_leaves_no_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            shutil.copytree(ROOT / "fixtures/tiny-prenorm", source)
            model = bytearray((source / "model.bin").read_bytes())
            model[0] ^= 1
            (source / "model.bin").write_bytes(model)
            out = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "integrity"):
                quantize(source, out)
            self.assertFalse(out.exists())
            self.assertFalse(list(Path(tmp).glob(".int8-pack-*")))

    def test_candid_job_id_escapes(self):
        self.assertEqual(decode_blobs(r'job = blob "\00\ff\\\"\n"'), [bytes([0, 255, 92, 34, 10])])

    def test_upload_length_encoding(self):
        self.assertEqual(uleb(0), b"\0")
        self.assertEqual(uleb(127), b"\x7f")
        self.assertEqual(uleb(128), b"\x80\x01")
        self.assertEqual(uleb(1024*1024), b"\x80\x80\x40")

    def test_head_activation_is_not_encoder_activation(self):
        upstream = dict(vocab_size=50368, hidden_size=1024, num_hidden_layers=28,
                        num_attention_heads=16, intermediate_size=2624, norm_eps=1e-5,
                        global_attn_every_n_layers=3, local_attention=128, hidden_activation="gelu",
                        rope_parameters={"full_attention": {"rope_theta": 160000}, "sliding_attention": {"rope_theta": 10000}})
        self.assertEqual(translate_config(upstream, {"head_layers": 2})["decision_activation"], "Relu")

class ProfileReplyTests(unittest.TestCase):
    def test_profile_costs_parse_nested_shape_without_progress(self):
        from canister_infer import parse_costs
        reply = '(variant { Ok = record { costs = vec { record { shape = vec { 128 : nat64; 3072; 1024 }; instructions = 1_234 : nat64; name = "int8.matmul" }; record { name = "norm"; shape = vec { 0; 0; 0 }; instructions = 42 } }; progress = record { instructions = 9999 } } })'
        self.assertEqual(parse_costs(reply), [dict(name="int8.matmul",shape=[128,3072,1024],instructions=1234),dict(name="norm",shape=[0,0,0],instructions=42)])


if __name__ == "__main__":
    unittest.main()
