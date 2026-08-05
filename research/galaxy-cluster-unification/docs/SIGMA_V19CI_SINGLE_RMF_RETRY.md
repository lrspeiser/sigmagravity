# Sigma V19CI single-RMF retry

V19CD repaired the missing-CIAO environment and then completed 383 of the 384
missing response cells.  One cell, `BULLET_bin384_obs5358_ccd0`, failed inside
`specextract` while creating its RMF.  The source PHA, background PHA and ARF
were created in the partial directory, but no completed checkpoint exists.

V19CI authorizes exactly one operational retry of that absent checkpoint.  It
does not modify the response method or any physics setting.

The retry must:

- begin only after every V19CD/V19W5 process has exited;
- verify 383 completed checkpoint reports and the one exact failed partial;
- verify the failed log hash and RMF-creation signature;
- move the partial directory into `failed_attempts` rather than delete it;
- run the byte-identical V19W5 config and runner in the same verified CIAO
  environment;
- reuse the 383 completed checkpoints and regenerate only the absent completed
  checkpoint;
- pass the final audit of 5,082 cells, 20,328 products, and the byte-identical
  protected base archive;
- resume the unchanged V19BR source-only chain only after that pass.

A second retry is not automatic.  No lensing, halo, action, gravity-parameter,
or holdout target is opened by this protocol.

Preflight from WSL:

```bash
python scripts/run_sigma_v19ci_single_rmf_retry.py --preflight-only
```

Execution uses `--execute` only after the config, runner hash, and tests are
committed.

## Outcome

The authorized retry reproduced the same failure for the same cell with no
other response cell executing concurrently.  Both preserved attempts have the
same `specextract.log` SHA-256,
`ee083a7ff3e68d11498a68ede5a111c92b518907c3e10dd1fcbd84989a72933c`,
and both stop at `Failed to create RMF`.  The scratch still contains 383 valid
completed checkpoints; the first partial is preserved under
`failed_attempts/c3432_rmf_attempt1`, and the second partial remains available
for diagnosis.

V19CI therefore failed closed.  V19BR was not started.  Another blind retry is
not authorized; the next step must diagnose the RMF boundary with higher CIAO
verbosity in a non-admitted copy before any response-placement remedy is
defined.
