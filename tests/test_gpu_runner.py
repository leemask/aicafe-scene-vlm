"""Control-flow tests with simulated CUDA absence; no GPU success claims."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class GpuRunnerTests(unittest.TestCase):
    def load_runner(self):
        # Only isolate imports; test the actual runner and its failure records.
        core = types.ModuleType("app.core")
        core.PROMPT = "test"
        core.normalize = lambda x: x
        core.atomic_json = lambda p,d: Path(p).write_text(json.dumps(d))
        sampling = types.ModuleType("app.sampling")
        sampling.sample_frames = lambda *a: []
        spec = importlib.util.spec_from_file_location("gpu_runner_test", ROOT/"scripts/compare_vlm.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.core":core, "app.sampling":sampling}):
            spec.loader.exec_module(module)
        return module

    def test_incompatible_modes_rejected(self):
        module = self.load_runner()
        for device,dtype in [("cpu","float16"),("cuda","int8-dynamic")]:
            with self.subTest(device=device), patch.object(sys,"argv",["runner","--run-id","test","--device",device,"--dtype",dtype]):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code,2)

    def test_cuda_absence_records_failure_before_download(self):
        module = self.load_runner()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda:False)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda *a,**k: self.fail("download must not start")
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = transformers.Qwen3VLForConditionalGeneration = object
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ["scripts/compare_vlm.py","app/core.py","app/sampling.py"]:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("fixture")
            (root/"samples").mkdir()
            (root/"samples/manifest.json").write_text('{"clips":[{"id":"a"}]}')
            with patch.object(module,"ROOT",root), patch.object(sys,"argv",["runner","--run-id","test","--device","cuda","--dtype","float16"]), patch.dict(sys.modules,{"torch":fake_torch,"huggingface_hub":hub,"transformers":transformers}), patch("importlib.metadata.version",return_value="test"):
                with self.assertRaisesRegex(RuntimeError,"CUDA unavailable"):
                    module.main()
            record=json.loads((root/"experiments/test/run.json").read_text())
            self.assertEqual(record["status"],"failed")
            self.assertEqual(record["completed_inferences"],0)
            self.assertFalse((root/"experiments/test/summary.json").exists())

if __name__=="__main__":
    unittest.main()
