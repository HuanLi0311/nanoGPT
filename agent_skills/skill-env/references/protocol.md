# Environment protocol

Run the read-only probe before a remote experiment:

```bash
python3 scripts/probe.py --host air-node-03 --output runs/<run>/env.json
```

The JSON records the exact SSH command, host identity, GPU inventory, Python,
package availability, disk line, and any SSH error. A nonzero `ssh_exit_code`
is an environment failure, never a model result. `air-node-03` is the only
allowed compute host for this project.
