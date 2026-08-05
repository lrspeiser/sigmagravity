# V19CN fine-WMAP final recovery

V19CN turns the successful V19CM detector-edge diagnostic into one final production checkpoint. It preserves both ordinary failed attempts, regenerates the exact manifest cell from its original event filters, and changes only `binwmap=det=8` to `binwmap=det=1`. All weighting, calibration, response position, source/background selection, energy ranges, and original V19W2 product audits remain mandatory.

After the cell passes, the byte-identical V19W5 runner independently validates the complete 5,082-cell, 20,328-product response archive and the protected base tree. Only that pass permits the already frozen V19BR source-only chain to resume. V19CN does not run V19BS, derive an action, open a lensing or halo target, change gravity constants, or optimize the Solar System.
