# V19CO checkpoint-attempt metadata correction

The V19CN response products passed their original physical and numerical gates. Its wrapper then incorrectly changed the successful producer's `attempt` field from `1` to `3` to reflect two earlier failed launches. The independent archive schema defines this field as the checkpoint producer's bounded attempt, while failed launches are recorded separately. It correctly rejected `3`.

V19CO binds the failed report, checkpoint, and four product hashes before changing anything. It changes only the JSON field `attempt: 3` to `attempt: 1`, retains the explicit edge-remediation and both failed-attempt paths, proves that no other top-level value or product byte changed, and reruns the independent checkpoint, full V19W5, and target-sealed V19BR audits. It cannot run V19BS or alter physics.
