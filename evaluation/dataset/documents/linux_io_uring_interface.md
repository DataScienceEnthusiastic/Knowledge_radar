# Linux io_uring Asynchronous System Call Interface

io_uring provides zero-copy, asynchronous I/O submission and completion via two lockless ring buffers: Submission Queue (SQ) and Completion Queue (CQ). Applications write I/O requests into the SQ and notify the kernel with an io_uring_enter system call (or use SQPOLL kernel polling). The kernel processes submissions asynchronously and posts completion entries to the CQ buffer, which user-space reads without syscall overhead. Registered buffers and files avoid per-operation buffer mapping and file descriptor reference counting.
