import os
import time
import warnings
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from model import AudioSpectrogramTransformer

warnings.filterwarnings("ignore")

def profile_pytorch(model: torch.nn.Module, dummy_input: torch.Tensor, num_runs: int = 150) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    return ((time.perf_counter() - start) / num_runs) * 1000.0


def profile_onnx(onnx_path: str, dummy_np: np.ndarray, num_runs: int = 150) -> float:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(20):
        _ = session.run(None, {input_name: dummy_np})

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: dummy_np})
    return ((time.perf_counter() - start) / num_runs) * 1000.0


def export_and_quantize(model, dummy_spec, prefix="ast"):
    fp32_onnx_path = f"saved_models/{prefix}_fp32.onnx"
    preproc_onnx_path = f"saved_models/{prefix}_preproc.onnx"
    int8_onnx_path = f"saved_models/{prefix}_int8.onnx"

    model.eval()
    export_kwargs = {
        "input_names": ["mel_spectrogram"],
        "output_names": ["class_logits"],
        "opset_version": 14,
        "do_constant_folding": True,
    }
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy_spec, fp32_onnx_path, **export_kwargs)

    try:
        quant_pre_process(
            input_model_path=fp32_onnx_path,
            output_model_path=preproc_onnx_path,
            skip_symbolic_shape=True
        )
        quant_input = preproc_onnx_path
    except Exception:
        quant_input = fp32_onnx_path

    quantize_dynamic(
        model_input=quant_input,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
        extra_options={"DisableShapeInference": True}
    )
    return fp32_onnx_path, int8_onnx_path


if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    # High-resolution 5s Mel-Spectrogram: [Batch=1, Channels=1, Mel_Bins=64, Time_Frames=800]
    # Produces N = (64/8) * (800/10) = 640 patches (641^2 = 410,881 attention pairs)
    high_res_spec = torch.randn(1, 1, 64, 800, dtype=torch.float32)
    high_res_np = high_res_spec.numpy()

    # Standard 5s Mel-Spectrogram: [1, 1, 64, 200] -> N = 160 patches
    std_spec = torch.randn(1, 1, 64, 200, dtype=torch.float32)
    std_np = std_spec.numpy()

    # --- REGIME A: Standard Resolution (N = 160 tokens) ---
    base_160 = AudioSpectrogramTransformer(num_classes=10, keep_ratio=1.0)
    pruned_160 = AudioSpectrogramTransformer(num_classes=10, keep_ratio=0.6)
    if os.path.exists("saved_models/ast_keep_100.pt"):
        base_160.load_state_dict(torch.load("saved_models/ast_keep_100.pt", map_location="cpu", weights_only=True))
    if os.path.exists("saved_models/ast_keep_60.pt"):
        pruned_160.load_state_dict(torch.load("saved_models/ast_keep_60.pt", map_location="cpu", weights_only=True))

    fp32_path, int8_path = export_and_quantize(pruned_160, std_spec, prefix="ast_160")
    ms_base_160 = profile_pytorch(base_160, std_spec)
    ms_onnx_fp32_160 = profile_onnx(fp32_path, std_np)
    ms_onnx_int8_160 = profile_onnx(int8_path, std_np)

    # --- REGIME B: High-Resolution Sequence Scaling (N = 640 tokens) ---
    base_640 = AudioSpectrogramTransformer(num_classes=10, keep_ratio=1.0)
    pruned_640 = AudioSpectrogramTransformer(num_classes=10, keep_ratio=0.5)
    # Resize positional embeddings for 640 patches
    base_640.pos_embed = torch.nn.Parameter(torch.randn(1, 641, 128) * 0.02)
    pruned_640.pos_embed = torch.nn.Parameter(torch.randn(1, 641, 128) * 0.02)

    fp32_640_path, int8_640_path = export_and_quantize(pruned_640, high_res_spec, prefix="ast_640")
    ms_base_640 = profile_pytorch(base_640, high_res_spec)
    ms_pruned_640 = profile_pytorch(pruned_640, high_res_spec)
    ms_onnx_int8_640 = profile_onnx(int8_640_path, high_res_np)

    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    int8_size_kb = os.path.getsize(int8_path) / 1024.0

    print("\n================ CPU INFERENCE & QUANTIZATION BENCHMARK ================")
    print("[Regime 1: Standard Spectrogram (N = 160 -> 96 Tokens, -63.7% Attention Pairs)]")
    print(f"  Baseline Unpruned AST (PyTorch FP32) : {ms_base_160:.2f} ms")
    print(f"  Spectral-Pruned AST   (ONNX FP32)    : {ms_onnx_fp32_160:.2f} ms ({ms_base_160/ms_onnx_fp32_160:.2f}x speedup)")
    print(f"  Spectral-Pruned AST   (ONNX INT8)    : {ms_onnx_int8_160:.2f} ms ({ms_base_160/ms_onnx_int8_160:.2f}x speedup)")
    print("\n[Regime 2: High-Res Spectrogram (N = 640 -> 320 Tokens, -75.0% Attention Pairs)]")
    print(f"  Baseline Unpruned AST (PyTorch FP32) : {ms_base_640:.2f} ms")
    print(f"  Spectral-Pruned AST   (PyTorch FP32) : {ms_pruned_640:.2f} ms ({ms_base_640/ms_pruned_640:.2f}x speedup)")
    print(f"  Spectral-Pruned AST   (ONNX INT8)    : {ms_onnx_int8_640:.2f} ms ({ms_base_640/ms_onnx_int8_640:.2f}x speedup)")
    print("------------------------------------------------------------------------")
    print(f"Model Artifact Compression : {fp32_size_kb:.1f} KB (FP32) -> {int8_size_kb:.1f} KB (INT8) ({fp32_size_kb/int8_size_kb:.2f}x smaller)")
    print("========================================================================")