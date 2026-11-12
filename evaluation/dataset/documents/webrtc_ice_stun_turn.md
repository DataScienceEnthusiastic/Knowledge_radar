# WebRTC Media Transport: ICE, STUN, and TURN Protocols

Interactive Connectivity Establishment (ICE) resolves NAT traversal for peer-to-peer real-time media streams. STUN (Session Traversal Utilities for NAT) allows clients to discover their public reflexive IP addresses and ports. When symmetric NAT prevents direct P2P connections, ICE falls back to TURN (Traversal Using Relays around NAT) servers to relay RTP packets. Media encryption is mandated via SRTP (Secure Real-time Transport Protocol), negotiated using DTLS handshake.
