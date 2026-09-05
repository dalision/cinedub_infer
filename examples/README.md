# CineDub demo examples

Three single-line JSONL rows, one per task mode. Each row is routed to a
different generation path by the two sentinel strings `trans_cap` and
`text_prompt` (see the top-level `README.md` for the full meta-token
cheatsheet).

| File | Mode | `trans_cap` | `text_prompt` | `ref_wav` |
|---|---|---|---|---|
| `v2a.jsonl`  | V2A (SFX only)             | `"none_speech"`                    | real SFX caption | omitted |
| `v2s.jsonl`  | V2S (dubbed speech, single or dual speaker) | narrated transcript with `<S>...<E>` | `"clean speech"` | omitted (default voice) |
| `v2sa.jsonl` | V2SA (joint speech + SFX)  | narrated transcript with `<S>...<E>` | real SFX caption | omitted (default voice) |

The referenced mp4s live in `examples/assets/`. They are trimmed clips
from the CineDub demo page (<https://cinedub2026.github.io/>) — the same
source used for the paper. CLIP + SynchFormer features are NOT shipped —
they are extracted online from each mp4 by default.

> **Note.** These three rows are a **minimal, self-contained smoke test**.
> The full set of demos from the paper (video-to-audio, dialogue dubbing,
> joint speech + SFX, plus the ablation grids and baseline comparisons)
> is hosted on the project page:
>
>     https://cinedub2026.github.io/

## What each row demonstrates

- **`v2a.jsonl` — V2A / basketball dribbling.** A 7-second VGGSound-style
  clip. `trans_cap="none_speech"` disables the speech branch; the DiT is
  driven only by video features and the SFX caption.
- **`v2s.jsonl` — V2S / two-speaker jukebox dialogue.** A 6.5-second
  movie clip with a 4-turn back-and-forth between two speakers.
  `text_prompt="clean speech"` disables the SFX branch, and the model
  handles both speakers in a single pass without any per-speaker
  `ref_wav` or diarization input (default-voice mode).
- **`v2sa.jsonl` — V2SA / Krishna cartoon (joint speech + SFX).** A
  10-second animated clip with two child speakers *and* a cartoon
  sound-effect sting. Both branches are active: `trans_cap` carries the
  dialogue and `text_prompt` carries the ambient / SFX caption.

## Running

```bash
python inference.py --task v2a  --input examples/v2a.jsonl  --output ./out/v2a
python inference.py --task v2s  --input examples/v2s.jsonl  --output ./out/v2s
python inference.py --task v2sa --input examples/v2sa.jsonl --output ./out/v2sa
```

Or the shell shortcut:

```bash
bash scripts/demo_infer.sh v2a
bash scripts/demo_infer.sh v2s
bash scripts/demo_infer.sh v2sa
```

Override the checkpoint or output dir with env vars — see
`scripts/demo_infer.sh` for the full list.

## Building your own rows

- **Zero-shot voice cloning.** Add `"ref_wav": "spk.wav"` (single-speaker)
  or `"ref_wav": ["spkA.wav", "spkB.wav"]` (dual-speaker) to a V2S / V2SA
  row and CineDub will clone the reference voice(s) instead of using the
  default one. The three shipped rows deliberately omit `ref_wav` to
  showcase the default-voice behaviour.
- **Caption prompts.** `examples/prompts/README.md` has the recipe for
  turning a raw ASR transcript into a well-formed `trans_cap`. The two
  shipped prompts (`singlespeaker_caption.txt` and
  `multispeaker_caption.txt`) are drop-in for Gemini 2.5 Pro / GPT-4o /
  Claude.
