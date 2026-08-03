# Exact field-model confirmation milestone

Date: 2026-08-02

## Outcome

The generic simulator no longer treats a free Boolean as sufficient evidence
that a researcher accepted an interpreted equation. Validation and execution
are now separate states:

1. `POST /api/v1/models/validate` checks the draft and returns its canonical
   manifest and exact computational `modelSha256`.
2. A structurally valid draft remains
   `awaiting_researcher_confirmation`.
3. The researcher inspects the canonical manifest and submits the returned
   hash plus the fixed acknowledgement to `POST /api/v1/models/confirm`.
4. The service returns a deterministic confirmation receipt and a model whose
   `source.confirmedModelSha256` is bound to that computation.
5. Field-job and batch preflight reject any unconfirmed model or any model
   changed after confirmation.
6. The Python worker repeats the exact hash check before numerical execution.

The acknowledgement is:

> I confirm that this canonical model is the equation I intend to execute.

The browser exposes this as a separate **Confirm exact hash** action. Merely
loading, pasting, translating, or validating a model does not trigger it. An
LLM may help create the draft or explain the audit, but must not silently
perform the acknowledgement for the researcher.

## Integrity properties

- Confirmation binds fields, equations, parameters, geometry, data
  requirements, observables, solver controls, and parameter policy.
- A change to any computational element changes `modelSha256` and makes the
  previous confirmation unusable.
- A mismatched expected hash returns `model_hash_changed` instead of silently
  accepting the newer document.
- The receipt is deterministic and content addressed; it contains no clock or
  server-local state.
- Confirmation is required even when a caller invokes the numerical worker
  directly rather than through the hosted preflight.

## What this proves

This closes the roadmap requirement that a translated mathematical expression
must be explicitly confirmed before execution. It prevents the service or an
LLM from silently changing the equation tree between review and computation.

## What this does not prove

Confirmation does not establish that a theory is physically correct, that its
boundary conditions are scientifically appropriate, or that the user is an
authenticated legal identity. Durable multi-user attribution and signed audit
logs still require the production database, authentication, and worker
infrastructure.

It also does not translate arbitrary prose or LaTeX into a model automatically.
The current safe path begins with a researcher- or tool-produced draft manifest,
validates it deterministically, and requires human confirmation of the exact
result.
