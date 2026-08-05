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
