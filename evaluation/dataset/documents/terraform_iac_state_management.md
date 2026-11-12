# Terraform Infrastructure as Code State Engine

Terraform tracks provisioned cloud infrastructure in a JSON state file (terraform.tfstate). The state file maps declarative HCL configurations to real-world cloud resource IDs and attributes. Remote backends (e.g. AWS S3 with DynamoDB state locking) prevent concurrent modification conflicts during terraform apply. Terraform builds a Directed Acyclic Graph (DAG) of resource dependencies to parallelize non-dependent resource creations.
