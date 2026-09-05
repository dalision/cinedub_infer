# exp/

Output directory for generation runs. The demo script (`scripts/demo_infer.sh`)
writes into `exp/demo/` by default; override with the `SAVE_DIR` env var if
you want a different location.

Everything under `exp/` is gitignored except this README and `.gitkeep`, so
your generated media, logs, and per-run checkpoints stay out of version
control.

## Auto-created subdirs

Each run of `scripts/demo_infer.sh <mode>` writes into
`exp/demo/<TASK>_cfg<cfg>_<steps>/` where `<TASK>` is:

- `VTA` for `v2a` mode (video → SFX)
- `VTS` for `v2s` and `v2sa` modes (video → dubbed speech; and speech + SFX)

The task prefix comes from the routing in `inference.py`
(`generate_multimodal_tasks`); `cfg` and `steps` mirror the
`--cfg` / `--steps` CLI args.
