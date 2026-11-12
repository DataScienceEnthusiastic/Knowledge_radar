# HTTP/3 and QUIC Transport Layer Architecture

HTTP/3 replaces TCP with QUIC, a transport protocol built on top of UDP. QUIC eliminates head-of-line blocking across multiplexed streams by managing per-stream flow control and independent packet acknowledgment. Connection migration allows seamless IP/port transitions (such as switching from Wi-Fi to cellular) using 64-bit Connection IDs. QUIC integrates TLS 1.3 encryption into its transport headers, encrypting packet numbers and control frames.
