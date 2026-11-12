# PostgreSQL Multi-Version Concurrency Control and Autovacuum

PostgreSQL implements MVCC using tuple header fields: xmin (creating transaction ID) and xmax (deleting/updating transaction ID). Updates create new row versions rather than in-place modifications, leaving dead tuples on table heap pages. Autovacuum scans pages to reclaim dead tuple storage space, update the free space map (FSM), and prevent transaction ID wraparound. Index scans consult the visibility map (VM) to perform Index-Only Scans when pages contain no dead tuples.
