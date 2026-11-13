import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstract_image_node import MODEL_PRESETS, AbstractImageNode, NodeSettings


class FakeImage:
    def __init__(self, width, height):
        self.size = (width, height)

    def save(self, path):
        Path(path).write_bytes(f"FAKE-{self.size[0]}x{self.size[1]}".encode())


class FakeOutput:
    def __init__(self, images):
        self.images = images


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt=None, negative_prompt=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None, generator=None,
                 callback_on_step_end=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        })
        if callback_on_step_end is not None:
            for step in range(num_inference_steps):
                callback_on_step_end(None, step, 0.0, {})
        return FakeOutput([FakeImage(width, height)])


class NodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pipeline = FakePipeline()
        self.node = AbstractImageNode(
            NodeSettings(output_dir=str(Path(self.tmp.name) / "out")),
            pipeline=self.pipeline,
        )

    def test_generate_slaat_op_en_retourneert_result(self):
        result = self.node.generate("paarse inktvlekken", width=512, height=512, seed=7)
        self.assertTrue(result.path.exists())
        self.assertEqual(result.path.parent.name, "out")
        self.assertEqual(result.seed, 7)
        self.assertEqual(result.image.size, (512, 512))
        self.assertEqual(result.width, 512)
        self.assertEqual(result.height, 512)

    def test_resolutie_wordt_afgerond_op_veelvoud_van_8(self):
        result = self.node.generate("x", width=513, height=300, seed=1)
        self.assertEqual((result.width, result.height), (512, 296))
        self.assertEqual(self.pipeline.calls[-1]["width"], 512)
        self.assertEqual(self.pipeline.calls[-1]["height"], 296)

    def test_snap_clamps_binnen_grenzen(self):
        self.assertEqual(AbstractImageNode._snap_size(10), 64)
        self.assertEqual(AbstractImageNode._snap_size(99999), 2048)
        self.assertEqual(AbstractImageNode._snap_size(520), 520)

    def test_system_prompt_wordt_vóór_prompt_gezet(self):
        self.node.update_settings(system_prompt="abstract, rauw")
        self.node.generate("inkt", seed=1)
        sent = self.pipeline.calls[-1]["prompt"]
        self.assertTrue(sent.startswith("abstract, rauw"))
        self.assertIn("inkt", sent)

    def test_leeg_system_prompt_geeft_alleen_prompt(self):
        self.node.update_settings(system_prompt="")
        self.node.generate("alleen dit", seed=1)
        self.assertEqual(self.pipeline.calls[-1]["prompt"], "alleen dit")

    def test_seed_min1_is_willekeurig_maar_geregistreerd(self):
        result = self.node.generate("x", seed=-1)
        self.assertGreaterEqual(result.seed, 0)
        self.assertLess(result.seed, 2**31)

    def test_progress_callback_krijgt_alle_stappen(self):
        stappen = []
        self.node.generate("x", seed=1, steps=5, progress_cb=lambda s, t: stappen.append((s, t)))
        self.assertEqual(stappen, [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])

    def test_leere_prompt_gooit_fout(self):
        with self.assertRaises(ValueError):
            self.node.generate("   ")

    def test_negative_prompt_wordt_meegegeven(self):
        self.node.generate("x", seed=1, negative_prompt="lelijk")
        self.assertEqual(self.pipeline.calls[-1]["negative_prompt"], "lelijk")

    def test_defaults_komen_uit_settings(self):
        self.node.update_settings(steps=9, guidance_scale=3.5)
        self.node.generate("x", seed=1)
        call = self.pipeline.calls[-1]
        self.assertEqual(call["num_inference_steps"], 9)
        self.assertEqual(call["guidance_scale"], 3.5)

    def test_settings_roundtrip(self):
        cfg = Path(self.tmp.name) / "config.json"
        self.node.update_settings(system_prompt="anders", steps=9)
        self.node.save_settings(cfg)
        loaded = NodeSettings.load(cfg)
        self.assertEqual(loaded.system_prompt, "anders")
        self.assertEqual(loaded.steps, 9)

    def test_from_config(self):
        cfg = Path(self.tmp.name) / "c.json"
        NodeSettings(model_id="abc/def", output_dir=str(Path(self.tmp.name) / "o2")).save(cfg)
        node = AbstractImageNode.from_config(cfg)
        self.assertEqual(node.settings.model_id, "abc/def")

    def test_update_settings_model_id_reset_pipeline(self):
        self.node.generate("x", seed=1)
        self.assertIsNotNone(self.node._pipeline)
        self.node.update_settings(model_id="ander/model")
        self.assertIsNone(self.node._pipeline)

    def test_onbekende_instelling_gooit_fout(self):
        with self.assertRaises(ValueError):
            self.node.update_settings(geen_instelling="ja")

    def test_cache_dir_roundtrip(self):
        cfg = Path(self.tmp.name) / "cfg.json"
        node = AbstractImageNode(
            NodeSettings(output_dir=str(Path(self.tmp.name) / "o"), cache_dir="models"),
            config_path=cfg,
        )
        node.save_settings()
        loaded = NodeSettings.load(cfg)
        self.assertEqual(loaded.cache_dir, "models")

    def test_model_presets_structuur(self):
        self.assertGreaterEqual(len(MODEL_PRESETS), 2)
        for label, preset in MODEL_PRESETS.items():
            self.assertIsInstance(label, str)
            self.assertTrue(preset["model_id"])
            self.assertIsInstance(preset["steps"], int)
            self.assertIsInstance(preset["guidance_scale"], (int, float))

    def test_build_prompt_combineert_delen(self):
        self.assertEqual(AbstractImageNode.build_prompt("b", "a"), "a, b")
        self.assertEqual(AbstractImageNode.build_prompt("b", ""), "b")
        self.assertEqual(AbstractImageNode.build_prompt("b", "  "), "b")


if __name__ == "__main__":
    unittest.main()
