# Apache Kafka Log Compaction and ISR Protocol

Kafka topics are divided into ordered partitions for horizontal scalability and parallel consumption. Each partition has one leader and zero or more followers. The In-Sync Replicas (ISR) set tracks followers caught up with the leader. The High Watermark (HW) indicates the latest message offset successfully replicated to all ISR members. Consumers can only read messages up to the High Watermark to prevent reading uncommitted data. Log compaction retains the latest message value for each key within a compacted topic partition.
