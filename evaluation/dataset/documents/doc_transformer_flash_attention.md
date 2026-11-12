# FlashAttention-3 Algorithmic Analysis
FlashAttention-3 leverages hardware-specific asynchronous memory operations on Hopper (H100/H800) GPUs.
Key optimizations include:
1. Producer-consumer asynchronous warpgroup execution utilizing Tensor Memory Accelerator (TMA).
2. Overlapping softmax computation with GEMM using FP8 matrix multiply units (WGMMA).
3. Ping-pong scheduling between Tensor Core operations and shared memory transfers.
FlashAttention-3 achieves up to 740 TFLOPs/s in FP16 and over 1.2 PFLOPs/s in FP8 on NVIDIA H100 GPUs, providing a 1.8x to 2.2x speedup over FlashAttention-2.
