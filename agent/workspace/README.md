# Workspace boundary

`workspace/` defines the file-level boundary shared by the TypeScript harness
and the Python/Verl adapter.

- `root` is the only task-owned file tree.
- Relative paths and `/workspace/...` are resolved below `root`.
- Absolute paths outside the root and symlink components are rejected.
- `snapshot` records evidence; it does not provide isolation.
- `exec_command` runs a host process with the workspace as `cwd`. It does not
  restrict network, `/proc`, capabilities, or arbitrary paths inside shell text.

The current execution mode is therefore `workspace_host`. An OS/container
sandbox may be added behind the executor later without changing the tool ABI.
