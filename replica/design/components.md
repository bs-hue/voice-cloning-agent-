# Components: SwaraLab

## Button
- variants: primary, secondary, ghost, danger
- sizes: sm 32px, md 40px, lg 48px
- states: default, hover, active, focus-visible (2px ring, accent), disabled, loading
- tokens: bg accent, text on-accent, radius md
- a11y: native `<button>`, visible focus ring, `disabled` + `aria-busy` during loading
- used on: S01, S02, S03, S04

## Input / Textarea
- variants: text, search, multiline
- states: default, focus, error, disabled
- tokens: bg surface-raised, border border-input, text text, radius md
- a11y: explicit `<label>` tied by `htmlFor`, inline error message with `aria-describedby`
- used on: S01, S02, S03

## AudioDropzone
- variants: file-upload
- states: idle, dragover, valid, invalid
- tokens: bg surface, border border-input dashed, radius lg
- a11y: keyboard-accessible file input trigger, clear accepted formats (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`)
- used on: S02

## AudioPlayer
- variants: compact, full
- states: idle, playing, paused, error
- tokens: bg surface, border border, accent accent
- a11y: native `<audio controls>` or labeled custom controls, accessible duration/scrub
- used on: S01, S02, S03

## ProgressBanner
- variants: synthesis
- states: idle, active, complete, error
- tokens: bg surface-raised, border border, accent accent / danger danger
- a11y: `role="status"` / `aria-live="polite"`, clear step label and elapsed seconds
- used on: S01, S04

