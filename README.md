# wow-alert

Real-time cast-bar awareness for World of Warcraft. A PySide6 desktop app captures
the WoW window, runs a YOLO detector for cast bars, OCRs the spell + target +
duration, matches against a curated spell database, and plays a "DANGER" alert
when a heinous spell is detected. Detected events stream into a live log pane.

## Requirements

- Windows 10/11
- Python 3.12
- CUDA-capable NVIDIA GPU (tested on RTX 3090)
- World of Warcraft running in **Windowed** or **Windowed Fullscreen (borderless)** mode

## Setup

```
poetry install
```

This installs the package and exposes the `wow-alert` console script.

First launch downloads the rapidocr ONNX models (~15 MB) and pre-renders the TTS
phrase set (e.g. `DANGER.wav`) into the OS user-data directory
(`%LOCALAPPDATA%\wow-alert\tts_cache\` on Windows). Subsequent launches reuse
both.

## Running

```
poetry run wow-alert
```

The app window opens with three panes:

1. Live annotated frame from the WoW window. Aspect ratio is preserved when you
   resize.
2. Controls: confidence slider, pause/resume, alerts on/off toggle.
3. Append-only log of every detected cast, every alert, and any pipeline errors.

### Flags

| Flag | Purpose |
|---|---|
| `--config PATH` | Path to `app.yaml` (default: `config/app.yaml`). |
| `--model PATH` | Path to YOLO weights (`.pt` or `.engine`). |
| `--window-title TITLE` | Exact window title to capture. |
| `--confidence FLOAT` | Detection confidence threshold. |
| `--imgsz INT` | YOLO inference resolution (long side). Must match the value used at training time. |
| `--dungeon NAME` | Scope the spell DB + counters to this dungeon (iter 2 auto-fills). |
| `--class NAME` | Player class for counter filtering (iter 2 auto-fills). |
| `--spec NAME` | Player spec for counter filtering (iter 2 auto-fills). |

### Env vars

| Env var | Equivalent |
|---|---|
| `WOW_ALERT_MODEL` | `--model` |
| `WOW_ALERT_WINDOW_TITLE` | `--window-title` |

Persist in your PowerShell `$PROFILE` for hands-off launches.

### Game-side requirements

- WoW must be running in **Windowed** or **Windowed Fullscreen (borderless)**.
  Modern retail WoW's "Fullscreen" is borderless under the hood and works.

## Spell database

```
config/
  dungeons/
    _global.yaml                  # cross-dungeon spells + rules
    windrunner_spire.yaml         # one file per dungeon
    mists_of_tirna_scithe.yaml
    ...
```

Filename is the dungeon's display name slugified
(`"Mists of Tirna Scithe"` → `mists_of_tirna_scithe.yaml`). When
calibration identifies the active dungeon, the loader picks
`_global.yaml` **plus** that dungeon's file — no others — and feeds
both into the spell DB and rule engine.

Each file:

```yaml
dungeon: "Windrunner Spire"        # display name (omit for _global.yaml)

spells:
  - id: spirit_bolt
    name: "Spirit Bolt"
    aliases: ["SpiritBolt", "SpiritE Bolt"]
    severity: danger                # danger | info | ignore
    phrase: DANGER                  # TTS phrase key
    duration: 2.5                   # optional; trusted over OCR if present
    counters:
      - {class: paladin, spec: holy, action: BOP}

rules: []                           # placeholder — Phase E populates
```

### Match logic

`rapidfuzz.token_set_ratio` on the spell name with threshold 85;
`max(token_set_ratio, partial_ratio)` against the roster for target
canonicalization with threshold 70. Either check missing the threshold
returns no match (fail-closed — no false alerts).

## Training a new model

Label in CVAT → export → train with ultralytics → drop the new weights into
`config/app.yaml` (or pass `--model`).

### 1. Export from CVAT

Format: **Ultralytics YOLO Detection 1.0**. If you uploaded a video, check
**Save images**. Unzip into a stable location, e.g.
`D:/wow_rl_addon/models/modelN/dataset/`.

### 2. Verify `data.yaml`

Use an absolute `path:`; ultralytics resolves relative `path:` against the cwd,
not against the yaml file's location.

```yaml
path: D:/wow_rl_addon/models/modelN/dataset
train: images/train
val: images/train

names:
  0: cast_bar
```

If CVAT didn't emit a `val:` (no jobs tagged `Subset = Validation`), pointing val
at train works while iterating; val metrics will be inflated.

### 3. Train

```
poetry run yolo detect train data=D:/wow_rl_addon/models/modelN/dataset/data.yaml model=yolo11s.pt epochs=100 imgsz=1280 project=D:/wow_rl_addon/models/modelN/runs/detect
```

Match `imgsz` between training and inference. Output lands at
`D:/wow_rl_addon/models/modelN/runs/detect/train/weights/best.pt`.

### 4. Point the app at it

```
poetry run wow-alert --model D:/wow_rl_addon/models/modelN/runs/detect/train/weights/best.pt --imgsz 1280
```

## Optional: TensorRT export

2–5× faster inference on NVIDIA cards:

```
poetry run yolo export model=path/to/best.pt format=engine half=True imgsz=1280
```

Produces `best.engine`, loadable by `wow-alert` identically to `.pt`.

## Other platforms / GPUs

Real-time inference (`wow-alert`) is Windows-only — it uses `win32gui` to locate
the WoW window. Training and TensorRT export work cross-platform.

Default `torch`/`torchvision` are pinned to **CUDA 12.1** wheels:

```toml
[tool.poetry.dependencies]
torch = {version = "^2.5.1+cu121", source = "pytorch_cu121"}
torchvision = {version = "^0.20.1+cu121", source = "pytorch_cu121"}

[[tool.poetry.source]]
name = "pytorch_cu121"
url = "https://download.pytorch.org/whl/cu121"
priority = "explicit"
```

To target a different stack, swap the source URL and re-lock. Examples:

| Target | Source URL |
|---|---|
| CUDA 12.4 | `https://download.pytorch.org/whl/cu124` |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| ROCm 6.2 (Linux + AMD) | `https://download.pytorch.org/whl/rocm6.2` |
| CPU-only (any OS) | `https://download.pytorch.org/whl/cpu` |
| macOS (MPS) | omit the source — `torch = "^2.5.1"` |

`pywin32` is already marked `sys_platform == 'win32'`, so it's skipped elsewhere.

## Layout

```
wow_alert/
  cli.py            # entry point: parse args, wire deps, launch UI
  config.py         # AppConfig (pydantic) + YAML loader
  events.py         # shared dataclasses + protocols
  capture.py        # find window by title + grab frames via mss
  detector.py       # YOLO wrapper
  tracker.py        # IoU-based cast-bar tracker (new vs continuing)
  ocr.py            # rapidocr (ONNX) wrapper
  cast_bar.py       # pure parser: OCR tokens -> CastEvent
  rules.py          # SpellDb + RuleEngine (fuzzy match, fail-closed)
  audio.py          # TTS prerender + winsound playback
  pipeline.py       # PipelineWorker: single-thread capture→detect→tracker→...
  ui/
    main_window.py
    frame_widget.py # preserves aspect ratio via fit_to_window
    log_widget.py
  tools/
    import_spells.py  # stub; hand-edit spells.yaml for now

config/
  app.yaml
  spells.yaml

%LOCALAPPDATA%\wow-alert\   # generated at runtime, outside the project tree
  tts_cache/                # TTS-rendered phrase WAVs
  calibration.yaml          # LLM-derived screen layout (iter 2+)

tests/               # pytest + pytest-qt
```

## Running tests

```
poetry run pytest tests/
```

Tests cover the parser, tracker, rule engine, config loader, audio player, and the
pipeline worker (via mocks — no real YOLO, OCR, or audio invoked).
