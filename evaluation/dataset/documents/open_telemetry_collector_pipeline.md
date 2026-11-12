# OpenTelemetry Collector Architecture

The OpenTelemetry (OTel) Collector processes telemetry data (traces, metrics, logs) through a vendor-agnostic pipeline. Pipelines consist of Receivers (ingesting OTLP, Jaeger, Prometheus data), Processors (batching, memory limiting, attribute redaction), and Exporters. The OTLP wire protocol uses Protocol Buffers over gRPC or HTTP/protobuf. Context propagation injects W3C TraceContext headers (traceparent, tracestate) across distributed microservice boundaries.
