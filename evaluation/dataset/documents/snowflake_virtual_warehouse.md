# Snowflake Multi-Cluster Shared Data Architecture

Snowflake decouples centralized shared storage (Amazon S3 / Azure Blob) from independent virtual compute warehouses. Table data is partitioned into immutable columnar micro-partitions (50MB to 500MB uncompressed) with min/max metadata pruning. Multiple compute clusters can query the same underlying micro-partitions simultaneously without resource contention. The Cloud Services layer coordinates query optimization, metadata management, transaction ACID guarantees, and security.
