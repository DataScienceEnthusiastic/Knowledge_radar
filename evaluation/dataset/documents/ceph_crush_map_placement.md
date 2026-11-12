# Ceph Distributed Storage and the CRUSH Algorithm

Ceph eliminates centralized metadata lookups by using the CRUSH (Controlled Replication Under Scalable Hashing) algorithm. Objects are mapped to Placement Groups (PGs), and CRUSH computes the target Object Storage Daemons (OSDs) based on cluster weight maps and hierarchy rules. Storage failure domains (rack, row, datacenter) are encoded in the CRUSH map to guarantee replicas are placed in distinct physical zones. Monitors (MONs) maintain cluster membership and state maps using Paxos consensus.
