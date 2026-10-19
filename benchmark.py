import time
import torch
from model import AudioSpectrogramTransformer

def profile_model(model, dummy_input, num_runs=200):
    model.eval()
    # Warmup CPU cache
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    avg_ms = ((time.perf_counter() - start) / num_runs) * 1000.0
    return avg_ms

if __name__ == "__main__":
    dummy_spec = torch.randn(1, 1, 64, 200)  # Single 5-second audio spectrogram

    # 1. Baseline Unpruned AST (161 tokens incl. CLS -> 161^2 = 25,921 attention pairs)
    baseline_model = AudioSpectrogramTransformer(num_classes=10, keep_ratio=1.0)
    baseline_ms = profile_model(baseline_model, dummy_spec)

    # 2. Spectral-Pruned AST (97 tokens incl. CLS -> 97^2 = 9,409 attention pairs -> 63.7% attention reduction!)
    pruned_model = AudioSpectrogramTransformer(num_classes=10, keep_ratio=0.6)
    pruned_ms = profile_model(pruned_model, dummy_spec)

    # 3. Spectral-Pruned + Dynamic INT8 Quantization on Linear Attention Layers
    quantized_pruned_model = torch.quantization.quantize_dynamic(
        pruned_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_ms = profile_model(quantized_pruned_model, dummy_spec)

    print("\n=== CPU INFERENCE LATENCY BENCHMARK (Single 5s Audio Clip) ===")
    print(f"1. Baseline Unpruned AST (FP32, 160 patches)       : {baseline_ms:.2f} ms")
    print(f"2. Spectral-Pruned AST   (FP32,  96 patches, -40%) : {pruned_ms:.2f} ms ({baseline_ms/pruned_ms:.2f}x speedup)")
    print(f"3. Spectral-Pruned AST   (INT8,  96 patches, -40%) : {quantized_ms:.2f} ms ({baseline_ms/quantized_ms:.2f}x speedup)")
    print(f"Self-Attention Matrix Complexity Reduction         : 63.7% fewer token-pair operations")