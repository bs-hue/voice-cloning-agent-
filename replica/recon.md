# Recon map: VoiceStudio (Local Voice Cloning Slice)

Scope: Single-user local text-to-speech, authorized voice cloning, saved voice profiles, playback, WAV export, and generation history in English and Hindi, exposing only controls supported by Chatterbox Multilingual. (Transcription deferred to a later phase; dubbing, voice design, mobile, and cloud hosting out of scope.)
For: Single user running locally on Windows 11 (RTX 3050 4 GB VRAM, 31.82 GB RAM, E: drive) under original branding (**SwaraLab**).
Date: 2026-11-11

## Sources

| # | source | URL | notes |
| --- | --- | --- | --- |
| 1 | GitHub README | https://github.com/debpalash/VoiceStudio | Public feature overview, local-first positioning |
| 2 | Feature catalog | https://github.com/debpalash/VoiceStudio/blob/main/docs/feature-catalog.md | Public list of TTS/ASR capabilities |
| 3 | Engine guides | https://github.com/debpalash/VoiceStudio/blob/main/docs/engines/README.md | Public engine requirements and hardware constraints |
| 4 | Model download docs | https://github.com/debpalash/VoiceStudio/blob/main/docs/downloading-models.md | Public progress reporting and Hugging Face download behavior |

## Core loop

Upload an authorized voice sample, save it as a reusable voice profile, enter English or Hindi text, generate real cloned speech locally with clear progress feedback, then play and download the `.wav` file.

## Screens

| ID | screen | route / how to reach | purpose | key components | states seen |
| --- | --- | --- | --- | --- | --- |
| S01 | Studio (Generate Speech) | `/` (primary nav tab) | Select a saved voice profile, choose English or Hindi, optionally adjust supported `exaggeration` / `cfg_weight` / `temperature`, generate, play, and export WAV | VoiceSelector, LanguageToggle, TextPromptArea, SupportedTuningPanel, GenerateButton, ProgressBanner, AudioPlayer, ExportButton | empty, loading-model, generating, ready, error |
| S02 | Voice Profiles | `/voices` (primary nav tab) | Upload an authorized `.wav`/`.mp3` clip, check speaker authorization confirmation, name and save profile, preview/delete saved profiles | AudioDropzone, ConsentCheckbox, ProfileForm, ProfileCard, AudioPlayer | empty, invalid-audio, consent-unchecked, saving, filled, error |
| S03 | Generation History | `/history` (primary nav tab) | Browse, search, filter, replay, download WAV, and delete past generations | SearchBar, LanguageFilter, HistoryRow, AudioPlayer, DeleteConfirm | empty, filtered-empty, filled, playing, missing-file |
| S04 | Engine & Storage Status | `/setup` (primary nav tab) | Inspect GPU/CPU status, model weight status on `E:`, `E:` free space, and device mode (`auto` / `cuda` / `cpu`) | HardwareCard, ModelStatusBadge, ModeToggle, DiskMeter | not-downloaded, ready, cpu-fallback, error |

## Flows

```
F01 Core cloning and speech loop (vertical slice)
    S02 upload authorized clip + check consent + save profile -> S01 select profile + pick en/hi + enter text + Generate -> S01 live progress -> S01 play & download WAV -> S03 appears in history
    happy path clicks: 6
    edge: consent unchecked, reference <3s or silent, empty text, long text, GPU OOM / CPU fallback, Hindi Devanagari input

F02 Reuse saved voice profile
    S01 select saved profile -> enter new text -> Generate -> play & download WAV
    happy path clicks: 3
    edge: profile deleted, missing reference file on disk

F03 Manage generation history
    S03 search or filter by language -> replay clip -> download WAV or delete item
    happy path clicks: 2
    edge: output WAV manually deleted from disk, empty search results

F04 Engine & storage status
    S04 inspect hardware & E: space -> switch auto/cuda/cpu device mode
    happy path clicks: 2
    edge: CUDA unavailable or OOM fallback to CPU
```

## Components

| component | variants | states | used on |
| --- | --- | --- | --- |
| Button | primary, secondary, ghost, danger | default, hover, focus-visible, disabled, loading | S01, S02, S03, S04 |
| Card | default, elevated, status | default | S01, S02, S03, S04 |
| Input / Textarea | text, search, multiline | default, focus, error, disabled | S01, S02, S03 |
| Select / SegmentedToggle | language, voice, device | default, open, selected, disabled | S01, S02, S03, S04 |
| AudioDropzone | file-upload | idle, dragover, valid, invalid | S02 |
| AudioPlayer | compact, full | idle, playing, paused, error | S01, S02, S03 |
| ProgressBanner | synthesis | idle, active, complete, error | S01, S04 |
| Badge | language, hardware, status | neutral, success, warning, danger | S01, S02, S03, S04 |

## Inferred data model

```
VoiceProfile  id, name, language_tag (en | hi | bilingual), reference_audio_path,
              duration_sec, sample_rate, consent_confirmed (bool), consent_note,
              created_at, updated_at
              evidence: S02 profile form, S01 voice selector
              confidence: high

GenerationJob id, voice_profile_id, voice_name_snapshot, text, language (en | hi),
              engine_id, device_used (cuda | cpu), exaggeration, cfg_weight, temperature,
              status (queued | running | completed | failed),
              progress_pct, status_message, error_message, output_audio_path,
              audio_duration_sec, elapsed_sec, created_at
              evidence: S01 generation flow, S03 history list
              confidence: high

AppSettings   id, preferred_device (auto | cuda | cpu), chunk_max_chars, updated_at
              evidence: S04 engine status
              confidence: high
```

Relationships: VoiceProfile 1-n GenerationJob (deleting a profile sets `voice_profile_id` null while keeping `voice_name_snapshot`).

## Feature matrix

See `replica/features.csv`. Must: 11, should: 2, could: 0, skip: 5.

## Out of scope (cannot or should not be cloned)

- VoiceStudio source code, Electron packaging, logos, copy, and branding
- Multi-engine 16-model hub and unsupported TTS parameters (speed, pitch, seed)
- Transcription (deferred to next phase), video dubbing, voice design, mobile apps, and cloud accounts/billing

## Size

Screens 4, flows 4, entities 3. Hard parts: Stage 1 hardware feasibility on 4 GB VRAM, E:-pinned caches, clean progress/error UX. Size: M.
