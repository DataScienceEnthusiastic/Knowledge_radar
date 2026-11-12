# Amazon DynamoDB Partitioning and Global Secondary Indexes

DynamoDB partitions data across storage nodes using a hash function applied to the Partition Key (PK). Items with identical partition keys are sorted by Sort Key (SK) in B-tree structures on individual partition servers. Global Secondary Indexes (GSIs) maintain an asynchronous replica of base table attributes with a different PK and SK. Read consistency is eventually consistent by default, with strongly consistent reads supported on base tables at 2x RCU cost.
