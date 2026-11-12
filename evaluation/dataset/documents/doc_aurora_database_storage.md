# Aurora Distributed Storage Subsystem Internals
The storage layer decouples compute from storage using a custom log-structured distributed virtual volume.
Database redo log records are replicated 6 ways across 3 Availability Zones (AZs), with 2 copies per AZ.
Write quorum requires 4 of 6 storage nodes (4/6 write quorum) to acknowledge log persistence.
Read quorum requires 3 of 6 storage nodes (3/6 read quorum) to reconstruct data in case of node failure.
Segment size for storage protection groups is 10 Gigabytes.
Crash recovery is distributed and instant because storage nodes continuously apply redo logs asynchronously without database-level crash recovery.
