# Workspace boundary

`workspace/` defines the file-level boundary shared by the TypeScript harness
and the Python/Verl adapter.

- `root` is the only task-owned file tree.
- Relative paths and `/workspace/...` are resolved below `root`.
- Absolute paths outside the root and symlink components are rejected.
- `snapshot` records evidence; it does not provide isolation.
- `workspace_host` runs a host process with the workspace as `cwd`; it does not
  restrict network, `/proc`, capabilities, or shell paths.
- `sandbox_backend=bwrap` runs commands and verifiers with only the workspace
  writable, read-only platform binaries/libraries, a cleared environment, and
  isolated network/PID namespaces.

Legacy manifests default to `workspace_host`; the four-stage synthesis pipeline
defaults to `bwrap`. Neither mode changes the tool ABI.
