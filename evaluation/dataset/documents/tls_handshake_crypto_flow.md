# TLS 1.3 Cryptographic Handshake Protocol

TLS 1.3 reduces handshake latency to 1-RTT (Round Trip Time) by combining key exchange and cipher negotiation in the ClientHello. Key exchange relies on Ephemeral Diffie-Hellman (ECDHE) over Curve25519 or P-256, ensuring Perfect Forward Secrecy (PFS). Static RSA key exchange and insecure legacy ciphers (CBC mode, RC4, MD5) are completely removed from the specification. 0-RTT Early Data allows clients to send application data in the first flight using pre-shared keys (PSK), with replay vulnerability tradeoffs.
