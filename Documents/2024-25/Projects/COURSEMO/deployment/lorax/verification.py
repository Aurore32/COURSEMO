# verification.py
import torch
from importlib.util import find_spec

def verify_kernels():
    checks = {
        # ExLlama Kernels
        "exllama_kernels": lambda: __import__('exllama_kernels'),
        
        # Flash Attention
        "flash_attn": lambda: __import__('flash_attn').__version__,
        "flash_attn.flash_attn_interface": lambda: __import__('flash_attn.flash_attn_interface'),
        
        # vLLM
        "vllm": lambda: __import__('vllm').__version__,
        "vllm._C": lambda: __import__('vllm._C'),  # Direct C++ extension check
        
        # AWQ
        "awq_inference_engine": lambda: __import__('awq_inference_engine'),
        
        # EETQ
        "eet": lambda: __import__('eet').__version__,
        
        # MegaBlocks
        "megablocks": lambda: __import__('megablocks').__version__,
        
        # Punica
        "punica_kernels": lambda: __import__('punica_kernels'),
        
        # Custom CUDA check
        "cuda_available": lambda: torch.cuda.is_available(),
        "cuda_arch_support": lambda: torch.cuda.get_device_capability()[0] >= 8  # Verify SM >= 80
    }

    failures = []
    for name, check in checks.items():
        try:
            result = check()
            print(f"✅ {name}: {result if not callable(result) else 'Verified'}")
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
            failures.append(name)

    if failures:
        raise RuntimeError(f"Critical kernel checks failed: {', '.join(failures)}")
    print("🎉 All kernel verifications passed!")

if __name__ == "__main__":
    verify_kernels()