# Memcached Slab Allocation Memory Manager

Memcached uses a slab allocation subsystem to prevent heap memory fragmentation caused by frequent malloc and free calls. Memory is pre-allocated into 1MB slab pages, which are subdivided into fixed-size chunk classes (e.g. 96 bytes, 120 bytes, 150 bytes). Item storage assigns values to the smallest chunk class capable of holding the key, value, and metadata flags. Eviction policies utilize a Least Recently Used (LRU) list per slab class when memory is exhausted.
