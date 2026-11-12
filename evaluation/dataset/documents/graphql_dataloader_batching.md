# DataLoader Batching and Caching Mechanism

DataLoader solves the GraphQL N+1 database query problem by batching individual key requests within a single Node.js event loop tick. Calls to dataloader.load(id) return promises and enqueue IDs in a batch queue. When the current execution tick finishes, DataLoader invokes a user-defined batch function with an array of all queued keys in a single SELECT WHERE id IN (...) query. Memoization caching deduplicates requests for identical keys across the request execution lifecycle.
