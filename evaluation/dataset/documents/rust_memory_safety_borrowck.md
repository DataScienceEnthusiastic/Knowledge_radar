# Rust Memory Safety and Borrow Checker Semantics

Rust guarantees memory safety without garbage collection through its ownership model. Every value has a single owner variable; when the owner goes out of scope, the value is dropped and deallocated. Borrowing rules mandate either any number of immutable references (&T) OR exactly one mutable reference (&mut T) at any time. The borrow checker verifies lifetimes at compile time, eliminating data races, use-after-free, and dangling pointer vulnerabilities.
