# WebAssembly (WASM) Binary Format and Execution Engine

WebAssembly is a portable stack-based bytecode format designed for near-native execution speed in web browsers and servers. WASM modules execute inside a linear memory space isolated from the host environment memory. The V8 engine compiles WASM bytecode using Liftoff (baseline fast compiler) and TurboFan (optimizing compiler). WASI (WebAssembly System Interface) defines standard POSIX-like system calls for sandboxed server-side execution.
