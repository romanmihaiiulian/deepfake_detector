# Attack lab: data collection, consent and rights

The lab generates face-swapped videos to measure the detector. Generating deepfakes of
real people is only acceptable with their informed, written consent. The tooling enforces
the record-keeping, but it cannot check the consent itself.

## Rules

1. **Written consent, per person, before any recording is used.** The consent must cover:
   - recording, storage and processing of the person's face video;
   - use of their face as a source *and* as a target of face swapping;
   - which uses are permitted (internal evaluation only, or also training), how long data
     is kept, and how to withdraw.
2. **Pseudonymous identities.** Register a person as `dfbench identity add <id>
   --consent-ref <ref>`. The ID must not be the person's name (use e.g. `p0007`). The
   consent reference points to the signed record, which is stored outside this
   repository, in your organisation's records system. Registration fails without it.
3. **Your own recordings only.** Use videos you recorded with the participant, or that you
   hold the rights to. Never use videos of people who have not consented, public figures
   included.
4. **Keep lab data apart.** Everything lives under `$DFD_DATA_ROOT` (default `./data`,
   gitignored, mode 0700). It never touches the API's `var/` directory or user uploads.
5. **Withdrawal.** `dfbench identity delete <id> --yes` deletes the person's source
   videos, intermediates, donor stills, every sample derived from them (as donor or as
   target), and their benchmark predictions and traces.
6. **Swapper licenses.** Most FaceFusion swapper models are non-commercial or research-only.
   Each sample records `swapper_license`. Only samples with a commercially usable swapper
   (currently `ghost_*`, Apache-2.0) may ever be considered for a commercial training set,
   and only after legal review.

## Recording guidance

Short clips of 20–30 s, one person, face visible most of the time, a little head movement
and talking. Vary the conditions across recordings: lighting, camera (laptop webcam, phone),
glasses, background. Interview-like footage is what matters for the HR use case.

Minimum participant numbers:

| Goal | Identities |
|---|---|
| Smoke test of the pipeline | 2 (one split) |
| Threshold tuning (val) + reporting (test) | 4 |
| Train, val and test (for the temporal model, step 14) | 6 or more; 30+ for meaningful numbers |

## Workflow

```sh
tools/install_facefusion.sh && export FACEFUSION_HOME=~/.local/share/deepfake-lab/facefusion
dfbench identity add p0001 --consent-ref CONSENT-2026-001
dfbench source add ~/recordings/p0001-a.mp4 --identity p0001
# ... more identities and sources ...
dfbench split
dfbench generate --dry-run      # see what would be produced
dfbench generate
```

Every generated fake is checked: the face must actually have changed in at least 80% of
sampled frames, or the sample is recorded as rejected.
