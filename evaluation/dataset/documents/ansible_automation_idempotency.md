# Ansible Automation Engine and Module Execution

Ansible is an agentless configuration management tool that executes automation playbooks over SSH or WinRM. Playbooks declare desired state in YAML; Ansible modules enforce idempotency by modifying target systems only when state diverges. Ansible copies standalone Python module code to remote host temp directories, executes the script, and parses JSON output. Facts gathering retrieves system hardware, network, and OS details to populate Jinja2 template variables.
