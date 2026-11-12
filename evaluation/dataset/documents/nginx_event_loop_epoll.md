# NGINX Event-Driven Asynchronous Architecture

NGINX operates on a non-blocking, event-driven architecture using a single master process and multiple worker processes. Each worker process handles tens of thousands of concurrent connections using OS event notification facilities like epoll (Linux) or kqueue (BSD). Worker processes do not spawn threads per connection, avoiding context-switch overhead. Upstream connection pooling keeps keepalive TCP connections open to backend application servers.
