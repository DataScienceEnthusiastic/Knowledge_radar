# ScyllaDB and Seastar Shared-Nothing C++ Architecture

ScyllaDB rewrites Apache Cassandra in C++20 using the Seastar asynchronous shared-nothing framework. Each CPU core runs an independent thread with its own memory pool, event loop, and network queue, eliminating lock contention. Disk I/O utilizes Linux asynchronous I/O (io_uring or libaio) directly to bypass OS page cache overhead. Dynamic memory management self-tunes cache sizes and compaction buffers to maintain sub-millisecond p99 latencies.
