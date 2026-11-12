# Linux Completely Fair Scheduler (CFS) Internals

The Linux Completely Fair Scheduler (CFS) allocates CPU execution time to tasks using a red-black tree indexed by virtual runtime (vruntime). Tasks with higher nice priority accrue vruntime at a slower rate than low-priority tasks. CFS always picks the leftmost task with the minimum vruntime for execution. Latency target scheduling guarantees every runnable process runs at least once within a targeted scheduling latency window.
