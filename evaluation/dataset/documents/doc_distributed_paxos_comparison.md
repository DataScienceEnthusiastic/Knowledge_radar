# Paxos vs. Raft Consensus Comparative Study
While Raft enforces strong leader invariants and contiguous log replication without holes, Multi-Paxos allows out-of-order proposal acceptance.
In Multi-Paxos, Phase 1 (Prepare/Promise) can be amortized across a sequence of log entries if the leader remains stable.
However, failure recovery in Multi-Paxos requires log hole resolution and phase reconciliation, making implementation complex.
Raft simplifies state space by prohibiting commits from previous terms unless an entry from the current term is also committed.
In throughput benchmarks under high network partition frequency, Raft recovered stable leader election in 220ms, whereas Multi-Paxos required 410ms to re-elect and reconcile holes.
