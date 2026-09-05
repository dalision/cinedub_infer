# Changelog

All notable changes to the CineDub inference release are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- README paper badge switched from "pending" to the arXiv preprint
  ([2608.15734](https://arxiv.org/abs/2608.15734)); project-page badge added; arXiv URL added to the BibTeX entry.
- `CITATION.cff` now carries the full seven-author list and the arXiv identifier
  (replaces the single-author placeholder used during the pre-release window).

## [1.0.0] - 2026-09-05

First public release.

### Added
- Repository visibility switched from private to public.
- `CHANGELOG.md` and a dated **News** section in `README.md`.
- GitHub release `v1.0.0` pinned to this commit so the code state can be cited reproducibly.

### Changed
- `exp/README.md` now points at `inference.py` (the old `pyscripts/generate_v2a_cond.py` path was removed when the layout was flattened).

### Notes
- One unified DiT checkpoint (`weights/cinedub/checkpoint/step=300000.ckpt`, hosted at
  [Dalision/cinedub](https://huggingface.co/Dalision/cinedub)) drives V2A, V2S and V2SA.

## [0.1.0] - 2026-07-31

Private preview. Inference-only extraction of the CineDub training codebase.

### Added
- Root-level `inference.py` CLI with `--input / --task / --seed / --ref_wav` shortcuts,
  auto-derived `id / dur / wav_path / feature`, and a strict per-task JSONL validator.
- Per-task CFG auto-default: V2A = 2.5, V2S / V2SA = 7.0.
- Muxed mp4 output (input video + generated audio) written next to the wav.
- Synchformer weights auto-downloaded from `hkchengrex/MMAudio`; `mmaudio` added as a pip dependency.
- Demo examples for four modes (`v2a`, `v2s`, `v2s_ap`, `v2sa`) with silent-video assets from
  <https://cinedub2026.github.io/>.
- Caption prompts for building `trans_cap` (single-speaker and multi-speaker) under `examples/prompts/`.
- README rewrite, `CITATION.cff`, MIT `LICENSE`, `environment.yml`, `.env.example`.

### Changed
- Repository layout flattened; task name `V2AS` renamed to `V2SA` to match the paper.
- Loader auto-derives `feature.clip / feature.sync` from `video_path` when a JSONL row omits them.

### Fixed
- Minimal 4-field example rows no longer crash the loader.
- `ref_wav` example in README respected the 3 s audio-prompt cap.
