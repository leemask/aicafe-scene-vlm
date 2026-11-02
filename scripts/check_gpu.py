"""Diagnose CUDA without downloading model weights."""
import json
import sys
import torch

def main():
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        x = torch.ones((64, 64), device="cuda", dtype=torch.float16)
        assert torch.isfinite(x @ x).all().item()
        torch.cuda.synchronize()
        info["fp16_matrix_test"] = "passed"
    print(json.dumps(info, indent=2))
    return 0 if info["cuda_available"] else 1

if __name__ == "__main__":
    sys.exit(main())
