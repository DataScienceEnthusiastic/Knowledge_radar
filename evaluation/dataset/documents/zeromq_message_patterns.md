# ZeroMQ Socket Patterns and Lockless Queues

ZeroMQ provides socket-like messaging abstractions without a dedicated broker daemon. Supported messaging topologies include Request-Reply (REQ-REP), Publish-Subscribe (PUB-SUB), and Push-Pull pipelines. ZeroMQ implements lock-free atomic queues (ypipe) for high-speed inter-thread message passing. Framing encodes length-delimited binary messages with multipart message support for envelope routing.
