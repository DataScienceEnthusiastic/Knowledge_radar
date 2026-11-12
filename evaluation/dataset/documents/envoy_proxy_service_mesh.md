# Envoy Proxy and Istio Service Mesh Architecture

Envoy is a high-performance C++ L4/L7 proxy deployed as a sidecar container alongside application workloads. The Dynamic Discovery Service (xDS) API allows dynamic configuration of listeners (LDS), routes (RDS), clusters (CDS), and endpoints (EDS). Envoy enforces automatic mTLS encryption, rate limiting, distributed tracing injection, and circuit breaking. Istio Pilot translates Kubernetes CRDs into Envoy xDS configuration streams over gRPC.
