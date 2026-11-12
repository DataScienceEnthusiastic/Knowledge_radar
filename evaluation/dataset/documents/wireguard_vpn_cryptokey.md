# WireGuard Secure Network Tunnel Protocol

WireGuard establishes encrypted Layer 3 VPN tunnels using Noise Protocol Framework primitives. Cryptography combines Curve25519 for ECDH key exchange, ChaCha20-Poly1305 for authenticated packet encryption, and BLAKE2s for hashing. Cryptokey Routing associates public keys directly with allowed IP addresses inside the kernel network device. Handshake occurs silently with 1-RTT, eliminating chatty heartbeat overhead when traffic is idle.
