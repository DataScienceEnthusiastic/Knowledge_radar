# Linux Namespaces and Cgroups Container Isolation

Linux containers leverage kernel namespaces for resource isolation: PID, Mount, Network, IPC, UTS, and User namespaces. Control Groups (cgroups v2) enforce resource limits on CPU time, memory consumption, I/O bandwidth, and process counts. Memory cgroups trigger the OOM killer when a container group exceeds memory.max without available swap. OverlayFS provides the copy-on-write (CoW) layered storage driver for container image layers.
