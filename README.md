# Audio Spectrogram Transformer (AST) with Spectral Token Pruning & ONNX INT8 Quantization

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005ced.svg)

An acoustic classification and CPU inference optimization pipeline built in **PyTorch**, **Torchaudio**, and **ONNX Runtime**. This repository mitigates the quadratic $O(N^2)$ self-attention bottleneck in Audio Spectrogram Transformers (AST) by introducing a **Hybrid Spectral Energy & Variance Gate** that dynamically prunes silent and ambient background time-frequency patches prior to Multi-Head Self-Attention.

---

## 1. Engineering Problem & Motivation

Standard Audio Spectrogram Transformers convert 1D audio waveforms into 2D Log-Mel Spectrograms, slice them into $N$ time-frequency patches, and compute full $O(N^2)$ self-attention across every token. In real-world environmental audio (such as the **ESC-50** benchmark), 35% to 55% of a 5-second recording consists of dead silence or steady-state background hum. Computing pairwise self-attention across uninformative background patches wastes CPU cycles and inflates inference latency.

This project solves that bottleneck on consumer x86_64 CPUs through a two-stage optimization pipeline:
1. **Algorithmic Complexity Reduction ($O(N^2)$ Token Pruning):** Dynamically scores and drops uninformative time-frequency tokens *before* the Transformer Encoder blocks, cutting attention matrix operations by **63.7% to 75.0%**.
2. **Hardware-Level Graph Compilation (ONNX INT8 Quantization):** Compiles the pruned PyTorch graph via TorchScript (`opset=14`) and applies dynamic `QInt8` weight quantization across linear attention projections to leverage Intel x86 vector instructions (`AVX2` / `VNNI`).

---

## 2. System Architecture & Mathematical Formulation

```text
Raw 5s Audio (.wav, 44.1kHz)
  │
  ▼  Downsample (16kHz) + Short-Time Fourier Transform (STFT)
2D Log-Mel Spectrogram [Batch, 1, 64 Mel Bins, T Frames]
  │
  ▼  Instance-Level Spectral Subtraction: (S - μ_t) / (σ + ε)
  │  (Cancels steady-state background noise across the time axis)
  │
  ▼  2D Conv Patch Projection + Pre-Pruning 2D Positional Encoding
Token Sequence [Batch, 1 (CLS) + N_patches, Embed_Dim=128]
  │
  ▼  ★ Hybrid Spectral Token Pruner: Score(x_i) = ||x_i||_2 + Var(x_i)
  │    Retains [CLS] + Top-K informative acoustic patches via torch.topk & torch.gather
Pruned Sequence [Batch, 1 (CLS) + K_patches, Embed_Dim=128]  (K = r · N)
  │
  ▼  4-Layer Transformer Encoder (Pre-LN, GELU, Multi-Head Self-Attention)
  │  + ONNX Runtime Dynamic INT8 Weight Quantization
  ▼
Linear Classification Head ([CLS] Token) ──► Predicted Acoustic Class Logits