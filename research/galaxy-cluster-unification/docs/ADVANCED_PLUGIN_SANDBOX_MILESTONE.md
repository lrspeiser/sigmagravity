# Signed advanced plug-in sandbox milestone

Date: 2026-08-03

## Outcome

The repository now has a separate advanced-code execution tier for stationary
models that do not fit the safe equation language. It does not add `eval`,
Python execution, or a Docker socket to Vercel or to the trusted field worker.
Public preflight can verify package authorship without running code; actual
execution requires a different host process, an operator trust record, and a
fresh isolated container.

The v1 plug-in ABI is deliberately small: one read-only `request.json` enters,
and one bounded `sigma-advanced-plugin-output/1` JSON document leaves on
stdout. Large binary artifact transport and a production package registry are
not part of this milestone.

## Package identity and trust

`sigma-advanced-plugin/1` binds:

- a semantic plug-in version and Python entrypoint;
- every package-relative path, byte count, and SHA-256;
- the exact Python 3.13.7, NumPy 2.2.6, and SciPy 1.16.1 ABI;
- CPU, memory, PID, wall-time, stdout, stderr, and temporary-space requests;
- the mandatory no-network/read-only/single-use isolation policy; and
- an Ed25519 publisher key identifier and detached signature.

The signature covers a domain-separated canonical document containing the
package hash. A valid signature means only that the holder of the corresponding
private key signed that manifest. It does not mean the publisher is approved,
the declared bytes are present, the code is benign, or the scientific result is
correct. Execution separately requires the key to be uniquely active in an
operator-controlled `sigma-plugin-trust-store/1` document.

Before container creation, the host resolves the package directory, rejects
links and special files, rejects missing or undeclared files, and rehashes every
declared byte. Production image references must include an allow-listed
`@sha256:` digest. A mutable image tag is accepted only behind an explicit CI
test flag.

## Container boundary

The host creates one named `--rm` container with:

- `--network none` and `--ipc none`;
- a read-only root filesystem;
- read-only package and dataset bind mounts;
- a fresh size-bounded `/tmp` tmpfs;
- fixed non-root UID/GID 65532;
- all Linux capabilities dropped and `no-new-privileges=true`;
- CPU-rate and CPU-time, memory/swap, PID, file-size, file-count, wall-time,
  stdout, and stderr limits;
- no forwarded environment variables or credentials; and
- no Docker socket or host output mount.

The only scientific output crosses the boundary as capped JSON stdout. The
trusted wrapper rehashes the package again inside the container, runs the
entrypoint in isolated Python mode with a minimal environment, caps both pipes,
kills the process group on violation, validates the output schema, and binds
the execution envelope to the package and input hashes. The host also removes
the exact named container after success, failure, timeout, or a killed Docker
client.

These controls use Docker's documented network, read-only root, resource, and
tmpfs mechanisms. Docker documents that `--read-only` restricts writes outside
explicit mounts and that tmpfs contents disappear when the container stops:

- <https://docs.docker.com/reference/cli/docker/container/run/>
- <https://docs.docker.com/engine/storage/tmpfs/>

## Acceptance fixture

The external fixture implements the same fixed simple-MOND acceleration as the
safe AST, but in uploaded-style Python. The acceptance runs it twice and
requires numerical agreement to relative tolerance `1e-12`. The plug-in also
reports directly observable sandbox facts:

- UID/GID are 65532;
- effective Linux capabilities are zero;
- `NoNewPrivs` is one;
- an IPv4 connection cannot be opened;
- writes to the dataset, package, and root filesystem fail;
- the Docker socket is absent;
- token, secret, password, and credential environment names are absent; and
- a `/tmp` sentinel from the first run is absent in the second run.

Changing one signed source byte is rejected before a container is created.
Unit acceptance additionally rejects a self-signed but untrusted publisher,
bad signatures, undeclared files, links, and unpinned production images. The
production research service returns
`advanced_plugin_registry_not_configured` instead of routing the job to the
safe worker.

## What remains before researcher use

This milestone is a real isolation fixture, not a multi-tenant hosted service.
Production still requires:

1. a project-scoped publisher enrollment and revocation workflow with audited
   human/operator approval;
2. safe archive ingestion and extraction with package-size, path, content-type,
   malware, and policy gates;
3. private content-addressed plug-in storage and an immutable package registry;
4. a dedicated sandbox host or stronger microVM/rootless boundary, never the
   Vercel process and never the trusted safe-language worker;
5. signed and vulnerability-scanned runtime images promoted by digest;
6. scheduler integration, per-project quotas, cancellation, result signing,
   monitoring, cleanup reconciliation, and abuse response;
7. a binary artifact ABI with a hard volume quota and complete manifest rehash;
8. adversarial escape, fork-bomb, memory, disk, timeout, malformed-output, and
   concurrent-run testing on the production kernel/runtime; and
9. scientific parity fixtures beyond the algebraic MOND example.

Container isolation reduces risk; it is not proof that arbitrary hostile code
is safe. Broad public access must wait for the dedicated-host and abuse-control
work above.
