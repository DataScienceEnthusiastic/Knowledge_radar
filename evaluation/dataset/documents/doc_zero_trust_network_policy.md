# Zero Trust Security Architecture (Policy Standard SEC-801)
Access control enforces continuous mutual TLS (mTLS) with SPIFFE/SPIRE cryptographic workload identity tokens.
Identity tokens have a maximum time-to-live (TTL) of 1 hour (3600 seconds) and must be rotated automatically.
Session authorization requires step-up multi-factor authentication (MFA) via FIDO2 WebAuthn hardware security keys.
All inter-service gRPC communication is encrypted with TLS 1.3 using AES-256-GCM or ChaCha20-Poly1305 cipher suites.
Network segmentation enforces default-deny policies at Layer 7 using eBPF kernel packet filtering.
