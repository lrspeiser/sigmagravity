# Generic field worker container

This image runs only the safe `sigma-field-model/1` equation language. It does
not execute uploaded Python. The later advanced-code tier requires a different,
single-use, network-disabled sandbox and must never share this trusted worker.

Build from the `research/galaxy-cluster-unification` directory:

```text
docker build --file worker/Dockerfile --tag sigma-field-worker:1.0.0-preview .
```

Package an input bundle:

```text
docker run --rm --network none --read-only \
  --mount type=bind,source=<absolute-work-directory>,target=/work \
  sigma-field-worker:1.0.0-preview \
  pack --arrays /work/input.npz --metadata /work/bundle-request.json --output /work/bundle
```

Run a job:

```text
docker run --rm --network none --read-only --memory 4g --cpus 2 \
  --mount type=bind,source=<absolute-work-directory>,target=/work \
  sigma-field-worker:1.0.0-preview \
  run --request /work/job-request.json
```

The process runs as non-root UID/GID 65532. Production scheduling must also set
a wall-time limit, PID limit, output-volume quota, and immutable input mount.
The output directory contains model, input, job, scientific result, residual
history, resource log, arrays, and artifact hashes. Wall time is not part of
the scientific result hash.
