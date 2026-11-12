# etcd v3 Architecture: Raft and bbolt KV Store

etcd is a distributed key-value store used as the backing datastore for Kubernetes clusters. etcd v3 represents data revisions monotonically, enabling MVCC historical watch operations and time-travel reads. The Raft consensus protocol handles replication, while data persistence is backed by bbolt, an embedded memory-mapped B+ tree KV engine. Keys are stored in bbolt with revision tuples (main, sub), allowing efficient range queries and compaction.
