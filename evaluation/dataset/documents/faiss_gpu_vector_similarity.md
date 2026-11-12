# FAISS GPU Vector Indexing Architecture

Meta FAISS accelerates dense vector nearest-neighbor search by parallelizing matrix multiplications on NVIDIA GPUs. Exact flat indexing calculates pairwise L2 distance or inner product using GPU cuBLAS SGEMM routines. For billions of vectors, FAISS IVF-PQ utilizes GPU Warp-level reduction to find nearest cluster centroids and decode quantized codes. Shared GPU memory caches centroid vectors to saturate GPU memory bandwidth during high-throughput batch queries.
