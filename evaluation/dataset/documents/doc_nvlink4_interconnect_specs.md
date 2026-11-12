# NVLink-4 and NVSwitch Architecture
NVLink Generation 4 provides 900 GB/s bidirectional bandwidth per NVIDIA H100 GPU (18 links per GPU, 50 GB/s per link).
The 3rd generation NVSwitch chip features 64 NVLink ports with an aggregate switching throughput of 3.2 Terabytes per second.
Hardware-based SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) accelerates distributed in-network reductions (AllReduce).
Compared to PCIe Gen5 (which offers 128 GB/s bidirectional), NVLink-4 offers a 7x bandwidth improvement for large language model tensor parallelism.
