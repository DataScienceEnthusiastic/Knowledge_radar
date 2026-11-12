# Kubernetes Kube-Scheduler Scoring and Filtering

Kube-scheduler assigns unscheduled pods to optimal nodes in two main phases: Filtering (Predicates) and Scoring (Priorities). Filtering eliminates nodes that do not satisfy pod resource requests, node selectors, taints, or affinity rules. Scoring ranks the remaining feasible nodes using plugins such as NodeResourcesBalancedAllocation and ImageLocality. The node with the highest aggregate score is selected and bound to the pod. Custom scheduling frameworks allow extending pre-filter, filter, pre-score, score, and bind phases via Go plugins.
