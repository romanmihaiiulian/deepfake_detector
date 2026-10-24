# Licensing policy

The project is licensed under **Apache-2.0** and must stay commercially usable. Every
dependency, model weight, dataset and tool is reviewed independently and recorded in
[`dependencies/licenses.yml`](../dependencies/licenses.yml).

## The inventory

The fields are `name`, `kind`, `version`, `source`, `license` (an SPDX expression or a
`LicenseRef-…`), `commercial_use` (`yes`, `no` or `review_required`), `distributed`,
`review_date` and `notes`.

`kind` is one of:

- `python` and `npm`: maintained by `tools/check_licenses.py --update`, so versions and
  `distributed` always match the installed environment and the web lockfile.
- `model`, `dataset`, `tool`, `system` and `reference`: written by hand.

`distributed: true` means the item ships with the product. For Python that is the
runtime dependency closure of `deepfake-core` and `deepfake-api`. For npm it is the
production dependencies that end up in the web bundle. Development tools, the benchmark
package and its dependencies are `distributed: false`.

## The gate

`make licenses` (part of `make check`) fails when:

- an installed package is missing from the inventory, has a different version, or an
  inventoried package is no longer installed;
- `distributed` disagrees with the computed value;
- a distributed item is not `commercial_use: yes`;
- a distributed item's license is copyleft (GPL, AGPL, LGPL, SSPL, EUPL),
  non-commercial, research-only, a RAIL license or unknown, and its notes carry no
  `EXCEPTION:` sign-off.

When a dependency is added or upgraded, run `uv run python tools/check_licenses.py
--update`. Then review every `review_required` entry: read the upstream license and set
`commercial_use`, `review_date` and `notes`. `--approve-permissive` handles the obvious
cases, where the license is purely MIT, BSD, Apache-2.0, ISC and similar.

## Rules that are easy to break

- **No code from non-commercial projects.** DeepfakeBench (CC BY-NC 4.0), DF40 and SBI
  may be read as research references. Their code must not be copied into the repository
  unless it is separately licensed.
- **Models are separate from code.** MIT-licensed weights can still come from
  research-only training data (GenD from FaceForensics++, YuNet from WIDER FACE). Those
  entries stay `review_required` until legal review.
- **FFmpeg.** PyAV's wheels bundle a GPL FFmpeg. Any binary we distribute must use an
  LGPL build ([ADR 0004](adr/0004-video-decoding-and-ffmpeg-licensing.md)).
- **The attack lab is internal.** FaceFusion (OpenRAIL-AS) and most of its swapper models
  (non-commercial or research RAIL) are used only to generate evaluation data and are
  never distributed. Each generated sample records the swapper's license, so that a
  commercially clean training subset can be selected.
- **Test fixtures** are fetched at test time and never committed when they are not
  public domain. For example, GenD's sample crops come from FF++ and Celeb-DF.
