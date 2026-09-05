# CineDub caption prompts

These prompts are used by an off-the-shelf multi-modal LLM (Gemini 2.5 Pro,
GPT-4o, Claude, …) to auto-generate the `trans_cap` field that CineDub
consumes at inference time. `trans_cap` is a composite string that fuses an
**objective acoustic description of each speaker** (identity, timbre, prosody,
environment) with the **verified spoken content wrapped in `<S>…<E>` tags**.

| File | When to use |
|------|-------------|
| `singlespeaker_caption.txt` | Diarized speaker count = 1 (single-speaker dubbing / long monologue) |
| `multispeaker_caption.txt`  | Diarized speaker count ≥ 2 (dialogue, panel, film scene) |

## Inputs the LLM needs

- The raw speech audio segment (attached as a multimodal input).
- A reference ASR transcription (Whisper, faster-whisper, whatever you use)
  substituted into `{transcription}` in the prompt.
- Optional: 3 sampled video frames if the target scene is visually
  informative (helps the LLM ground identity / environment cues).

## Expected output

A single composite string of the form:

```
[Acoustic desc 1] <S>[Spoken content 1]<E> [transition] [Acoustic desc 2] <S>[Spoken content 2]<E> …
```

Paste that string directly into the `trans_cap` field of your JSONL row (see
`examples/*.jsonl` for full-row templates).

## Notes

- Both prompts share the same "audio is king" rule: the LLM must trim
  hallucinated words at ASR boundaries and fill in truly-missing tokens by
  listening.
- `<S>` / `<E>` are literal tags used by the CineDub tokenizer to split
  utterances inside a shared cross-attention stream; do not swap them for
  other delimiters.
- Non-speech clips do **not** need a caption prompt — set `trans_cap` to the
  sentinel string `"none_speech"` and let the model take the V2A path (see
  `README.md (Meta-token cheatsheet section)`).
