-- SwaraLab local SQLite schema

create table if not exists voice_profiles (
  id text primary key,
  name text not null,
  language_tag text not null default 'bilingual'
    check (language_tag in ('en', 'hi', 'bilingual')),
  reference_audio_path text not null,
  duration_sec real not null check (duration_sec > 0),
  sample_rate integer not null,
  consent_confirmed integer not null check (consent_confirmed = 1),
  consent_note text not null default '',
  created_at text not null,
  updated_at text not null
);

create index if not exists idx_voice_profiles_created
  on voice_profiles (created_at desc);

create table if not exists generation_jobs (
  id text primary key,
  voice_profile_id text references voice_profiles(id) on delete set null,
  voice_name_snapshot text not null,
  text text not null,
  language text not null check (language in ('en', 'hi')),
  engine_id text not null default 'chatterbox-multilingual',
  device_used text not null default 'auto'
    check (device_used in ('auto', 'cuda', 'cpu')),
  status text not null default 'queued'
    check (status in ('queued', 'running', 'completed', 'failed')),
  progress_pct integer not null default 0
    check (progress_pct >= 0 and progress_pct <= 100),
  status_message text not null default '',
  error_message text,
  output_audio_path text,
  audio_duration_sec real,
  elapsed_sec real,
  created_at text not null,
  updated_at text not null
);

create index if not exists idx_generation_jobs_created
  on generation_jobs (created_at desc);

create index if not exists idx_generation_jobs_language
  on generation_jobs (language);

create table if not exists app_settings (
  id integer primary key check (id = 1),
  preferred_device text not null default 'auto'
    check (preferred_device in ('auto', 'cuda', 'cpu')),
  chunk_max_chars integer not null default 180
    check (chunk_max_chars >= 60 and chunk_max_chars <= 400),
  updated_at text not null
);

