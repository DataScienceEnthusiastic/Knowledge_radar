# OAuth 2.0 and JWT Token Security Standards

JSON Web Tokens (JWT) encapsulate JSON claims signed cryptographically using HMAC-SHA256 (HS256) or RSA/ECDSA (RS256/ES256). The OAuth 2.0 Authorization Code flow with PKCE (Proof Key for Code Exchange) protects public clients from authorization code interception. Access tokens should have short lifespans (5 to 15 minutes), while refresh tokens are stored in secure HTTP-only cookies with token rotation. Token revocation requires token introspection endpoints (RFC 7662) or distributed bloom filter denylists.
