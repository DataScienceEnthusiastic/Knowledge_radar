# gRPC and Protocol Buffers Binary Encoding

Protocol Buffers (Protobuf) serialize structured data into compact binary wire formats using variable-length varints and field tags. gRPC uses HTTP/2 transport to support bidirectional streaming, multiplexing, and flow control over a single TCP connection. Protobuf schema evolution allows adding optional fields without breaking backward or forward compatibility. gRPC interceptors provide middleware hooks for authentication, distributed tracing (OpenTelemetry), and logging.
