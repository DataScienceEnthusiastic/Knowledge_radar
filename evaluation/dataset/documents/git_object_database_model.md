# Git Content-Addressable Object Database

Git stores version control history in a content-addressable object store under .git/objects. Four primary object types exist: blob (file content), tree (directory structure and file names), commit (author, message, tree pointer), and tag. Objects are identified by the SHA-1 or SHA-256 hash of their header and compressed content. Packfiles combine multiple loose objects into a single compressed file using delta compression for efficient storage and network transfer.
