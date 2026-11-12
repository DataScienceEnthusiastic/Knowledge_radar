# Prometheus TSDB Block Storage and Gorilla Compression

Prometheus TSDB writes time-series metrics to 2-hour immutable disk blocks containing chunk files, index files, and metadata. Timestamps are compressed using double-delta encoding, reducing 64-bit timestamps down to 1-2 bits per sample. Float64 metric values are compressed using XOR floating-point compression derived from Facebook's Gorilla paper. Inverted indexes map series metric names and label sets to posting lists of 32-bit series IDs.
