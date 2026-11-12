# Redis Sentinel and Cluster High Availability

Redis Sentinel monitors master instances and orchestrates automatic failover when a master becomes unreachable. A master is declared Subjectively Down (sdown) when a sentinel cannot ping it for is-master-down-after-milliseconds. When a quorum of sentinels agrees, the master becomes Objectively Down (odown), triggering failover leader election via Raft-like voting. In Redis Cluster, data is partitioned across 16,384 hash slots using CRC16(key) mod 16384. Failover in Cluster mode is handled autonomously by gossip protocol among master nodes.
