# Attack lab and benchmark

The lab builds our own face-swap benchmark from consented recordings
([data-collection.md](data-collection.md)). The benchmark measures the detector on it,
with particular attention to the false-positive rate under conferencing conditions.

## Paired samples

Each consented source video A yields:

- one **real** sample: A itself;
- **fakes derived from A**: the face of a donor identity from the same split is swapped
  into A with FaceFusion, once per configured swapper.

All of these share `pair_id = A`. A detector that tells them apart cannot rely on
identity, background, camera or content. The only difference is the manipulation, so the
benchmark reports the **paired win rate**: how often a fake scores above its own source.

### Encoding parity

```
source A ──canonicalise──▶ A_canon (x264 crf 0, CFR ≤ 30 fps, ≤ 1080p, no audio)
                              ├──────────────── encode "pristine" ─▶ real sample
                              └─ FaceFusion (crf 0 output) ─▶ F_raw ─ encode "pristine" ─▶ fake sample
```

Both members of a pair go through the same final encode, and in M8 through the same
robustness variants. If reals kept their camera encoding while fakes were re-encoded, a
detector could learn codec history instead of manipulation.

### Coverage check

Swappers sometimes miss faces (profiles, small or fast-moving faces). For every fake, the
lab compares `F_raw` with `A_canon` frame by frame at 4 Hz. A frame counts as swapped when
the face region changed clearly more than the background. Fakes swapped in less than 80%
of sampled frames, or whose frame count differs from the source, are recorded as
`status: rejected` with a reason, so they are never mislabelled.

## Splits

Identities are split **before** any fake is made. Donors and targets are paired only
within a split, so no identity or source video can appear in two splits: no identity
leakage and no source-video leakage. A union-find check over identities and videos runs
after every generation. Splits open in priority order test, val, train, each only once it
can hold at least two identities. Held-out swappers (`held_out_for_test`) appear only in
test: the "unseen FaceFusion settings" slice. External datasets (`dfbench import`) form a
separate `external` split: the "unseen tools" slice.

## Provenance (`lab/registry/samples.jsonl`)

| Field | Meaning |
|---|---|
| `generator`, `generator_version` | `facefusion` and the installed version, `none` for reals, or the external dataset name |
| `swapper_model`, `swapper_license`, `generator_settings`, `generator_argv` | Exactly how the fake was made. Paths in `generator_argv` are reduced to file names |
| `source_video` | The video the sample derives from (also `pair_id`) |
| `source_identity` | The donor whose face was inserted |
| `target_identity` | The person in the source video, whose face was replaced |
| `resolution`, `fps`, `duration` | Of the final file |
| `transformation`, `transformation_argv`, `ffmpeg_version` | The encode or degradation profile, recorded exactly |
| `manipulation_coverage` | Fraction of sampled frames in which the face was swapped |
| `split`, `research_only`, `status`, `status_reason` | |

Identities are pseudonymous and carry only a `consent_ref`.
