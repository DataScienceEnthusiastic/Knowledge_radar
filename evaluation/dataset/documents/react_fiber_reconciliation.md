# React Fiber Reconciliation Engine and Virtual DOM

React Fiber is a complete rewrite of React's reconciliation engine that enables incremental rendering of the component tree. A Fiber node represents a unit of work containing component state, props, and pointers to child, sibling, and return fibers. The reconciliation algorithm operates in two phases: the interruptible Render phase (computing diffs) and the synchronous Commit phase (DOM mutation). Priority levels (Lane model) allow urgent updates (user typing) to interrupt lower-priority transitions (data fetching).
