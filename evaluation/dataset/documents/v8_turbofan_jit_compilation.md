# V8 Engine Ignition Interpreter and TurboFan JIT

The Google V8 JavaScript engine parses source code into an Abstract Syntax Tree (AST) and generates bytecode for the Ignition interpreter. Ignition collects inline cache (IC) profiling feedback during execution, tracking object property offsets and operand types. Functions that become hot are passed to the TurboFan optimizing compiler, which applies aggressive optimizations like function inlining and escape analysis. If type assumptions are violated during runtime, TurboFan executes deoptimization and bails back to Ignition.
