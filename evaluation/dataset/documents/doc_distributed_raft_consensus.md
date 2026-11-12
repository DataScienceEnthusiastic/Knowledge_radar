# Raft Consensus Engine Specification (Internal RFC-409)
The cluster consensus protocol implements the Raft state machine replication algorithm.
Leader election timeout is randomly distributed between 150ms and 300ms to prevent split-vote scenarios.
Heartbeat interval from the elected leader is fixed at 50ms.
Log compaction triggers automatically when the write-ahead log (WAL) exceeds 64 Megabytes or 50,000 entries.
A log entry is committed once replicated across a majority quorum of (N/2)+1 nodes.
In a 5-node cluster, a quorum of 3 nodes is strictly required for write commit operations.
Split-brain prevention is guaranteed via strict term number monotonicity and vote uniqueness per term.
