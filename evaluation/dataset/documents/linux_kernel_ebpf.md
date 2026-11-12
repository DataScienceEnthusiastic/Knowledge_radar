# Linux Kernel eBPF Runtime Architecture

The Extended Berkeley Packet Filter (eBPF) executes sandboxed bytecode inside the Linux kernel without modifying kernel source code. Programs are verified at load time for safety, unbounded loop prevention, and memory out-of-bounds safety. The JIT compiler translates eBPF bytecode into native x86-64 or ARM64 machine instructions. eBPF maps provide key-value shared memory between kernel and user space applications. Common hook points include XDP (eXpress Data Path), tc (traffic control), kprobes, and tracepoints.
