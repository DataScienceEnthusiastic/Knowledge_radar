# ClickHouse Columnar Storage Engine Internals

ClickHouse stores table data on disk ordered by primary key in MergeTree parts. Data is stored column-wise in compressed blocks (typically using LZ4 or ZSTD algorithms). Sparse primary indexes index the first row of every index granule (default 8,192 rows), allowing binary search page skipping. Background merge processes periodically combine smaller parts into larger immutable parts, executing mutations and TTL evictions.
