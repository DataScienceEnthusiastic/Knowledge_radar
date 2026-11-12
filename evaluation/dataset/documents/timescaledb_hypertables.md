# TimescaleDB PostgreSQL Hypertables and Chunking

TimescaleDB extends PostgreSQL for time-series data through partitioned hypertables. Hypertables automatically partition incoming time-series data into chunks based on time intervals and optional space keys. Queries against hypertables execute partition pruning to scan only chunks matching the query time window. Native columnar compression converts historical row-oriented chunks into compressed column-oriented storage.
