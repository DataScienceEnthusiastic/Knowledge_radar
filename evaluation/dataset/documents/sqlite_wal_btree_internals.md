# SQLite B-Tree and Write-Ahead Logging (WAL) Architecture

SQLite organizes database tables and indexes in disk pages formatted as B-trees (for tables) and B*-trees (for indexes). In Write-Ahead Logging (WAL) mode, modifications are appended to a separate -wal file instead of modifying database pages directly. Readers read pages from the main database file or the WAL index (wal-index) using shared memory, allowing concurrent readers and one writer. Checkpoints transfer modified pages from the WAL file back into the database file periodically or when the WAL reaches 1,000 pages.
