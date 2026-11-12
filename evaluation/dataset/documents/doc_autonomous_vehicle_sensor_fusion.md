# Autonomous Vehicle Sensor Fusion Pipeline (AV-Spec 3.1)
The perception stack fuses inputs from 8 high-resolution RGB cameras (8-megapixel, 30 fps), 5 automotive radar units (77 GHz), and 1 128-beam mechanical LiDAR.
LiDAR point clouds are voxelized at 0.1m resolution and processed through a 3D Sparse Convolutional backbone (SECOND/PointPillars).
Camera feature maps and LiDAR voxel features are unified in Bird's-Eye-View (BEV) representation space using cross-attention transformers.
Temporal latency from photon arrival to control command dispatch is strictly constrained to under 60 milliseconds.
Perception object tracking operates via an Extended Kalman Filter (EKF) with Hungarian bipartite matching at 50 Hz update frequency.
