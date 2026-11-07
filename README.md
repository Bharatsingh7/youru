# Youru 🎙️🔒
> **100% Offline Biometric Voice Assistant & Native Desktop Commander**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![faster-whisper](https://img.shields.io/badge/STT-faster--whisper%20turbo-FF6B6B?style=for-the-badge)](https://github.com/SYSTRAN/faster-whisper)
[![Silero VAD](https://img.shields.io/badge/VAD-Silero-4ECDC4?style=for-the-badge)](https://github.com/snakers4/silero-vad)
[![openWakeWord](https://img.shields.io/badge/WakeWord-openWakeWord-FFE66D?style=for-the-badge)](https://github.com/dscripka/openWakeWord)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**A private, zero-telemetry local voice assistant tailored for Hindi, English, and Hinglish. Features biometric voice lock authentication, custom wake-word gating, and deep Linux desktop system control.**

[✨ Highlights](#-key-features) • [🚀 Setup](#-setup-once) • [🎙️ First Run](#-first-run-step-by-step) • [🧠 Architecture](#-how-it-works) • [🛠️ Tools](#-desktop-tools-30)

</div>

---

## ✨ Key Features

- **Biometric Voice Lock**: TitaNet neural speaker embeddings verify against your enrolled voiceprint before executing commands; strangers get rejected.
- **Accent-Tuned Wake Word ("Youru")**: Custom openWakeWord ONNX acoustic model trained specifically for Indian English / Hindi accents, backed by a personal verifier.
- **Robust Speech Detection**: Silero VAD detects speech in real-time even under loud background fan noise and room acoustics.
- **Hindi & Hinglish STT**: Oriserve Hindi2Hinglish-Apex fine-tuned Whisper model for seamless mixed speech recognition.
- **Ultra-Fast Local Brain**: Local LLM execution via `llama.cpp` Vulkan server with hybrid GPU/CPU layer allocation (~79 tok/s, answers in 0.3–2.5 s).
- **Desktop System Controls (30 Tools)**: Media controls, window management, screen OCR, timers, reminders, volume, brightness, process management, and file inspection.

---

## 🚀 Setup (once)

```fish
cd ~/Projects/youru
uv sync --all-extras     # Python 3.12 venv with all dependencies
uv run youru download    # download base models (~160 MB); Whisper caches on first run
uv run pytest            # run test suite (163 offline tests)
```

## First run, step by step

Run everything in your own terminal (kitty or the VS Code terminal). The commands need your voice and keypresses.
Always start in the project folder:

```fish
cd ~/Projects/youru
```

`uv run` uses the project's Python automatically. You never need to activate a venv.

### Stage A — voice lock (about 10 min, works right away)

**Step 1. Mic check (1 min)**

```fish
uv run youru mic-check
```

1. It says *stay silent*. Press **Enter** and keep quiet for 5 s, with the fan running as usual.
2. It shows a sentence. Press **Enter** and read it aloud.
3. Read the result. The SNR should be green or yellow. If it says *Too loud* or *Quiet*, run the `wpctl` command it
   prints, then run mic-check again.

**Step 2. Record your voiceprint (5 min)**

```fish
uv run youru enroll
```

For each of the 20 sentences: press **Enter**, read it naturally, then stop talking. Recording stops by itself.

- Green **OK** means saved. Red **REDO** means the same sentence comes back; read it again.
- Some sentences ask you to lean back, turn the fan on or speak softly. Do it, because this makes the voice lock
  robust.
- `s` + Enter skips a sentence; `q` + Enter quits. At the end it builds your profile.

**Step 3. Try the voice lock (still using the "Hey Jarvis" placeholder)**

```fish
uv run youru listen --transcribe
```

- The first time, it loads Whisper for about a minute.
- Say **"Hey Jarvis, aaj ka weather batao"**. You should see **WAKE**, then **OWNER VERIFIED**, then the text it heard.
- Ask someone else to say the same thing. They should get **REJECTED**.
- Press **Ctrl+C** to stop.

**Step 4 (recommended). Let others try to fool it**

```fish
uv run youru record impostor --name mom      # repeat with each person's name
uv run youru eval --speaker                  # answer y to apply the thresholds
```

Each person reads 8 sentences. Without these, the voice-lock threshold is only estimated from your own voice.

### Stage B — the "Youru" wake word (~20 min recording + 1.5–3 h of Colab waiting)

Follow **[training/README.md](training/README.md)**. It walks through every click:

1. **Record:** `record wake`, `record confusables`, `train-pack`.
2. **Train in Google Colab:** uses a fixed notebook, because the official one is broken.
3. **Install on your PC:** drop in `youru.onnx`, run `record background`, `train-verifier` and `eval --wake`.

### Stage C — check your accent with speech-to-text (optional, 2 min)

```fish
uv run youru transcribe-test
```

It transcribes your enrollment sentences in auto, English and Hindi modes, and shows the word error rate for each,
so you can see which setting understands your Hinglish best. Set the winner in `config.yaml` under
`stt: language:` (`null` = auto, `"en"` or `"hi"`).

## Phase 2: the assistant (local LLM + tools + voice replies)

The voice lock is **switched off** for now (`speaker.enabled: false` in `config.yaml`), so anyone can talk to it.
Risky actions always ask first.

**One-time install** (llama.cpp's server, the LLM runtime):

```fish
sudo pacman -S llama-cpp ggml-vulkan
```

**Talk to it:**

```fish
uv run youru assist            # voice: say "Hey Jarvis, ..." or press Super+Y, then speak
uv run youru chat              # type instead (good for testing); add --speak to hear replies
```

Examples:
- "Hey Jarvis, what's the weather in Delhi?"
- "Hey Jarvis, aaj ki news batao"
- Super+Y, then "volume 30 percent"
- "open firefox"
- "play arijit singh on youtube"
- "how much battery is left?"
- "find my resume pdf"
- "what's on my clipboard?"
- "मौसम कैसा है?" (answers in Hindi)

| Part | What it uses |
|---|---|
| Brain | **Gemma 4 E2B** (Google's official 4-bit QAT build, 3.1 GB) on llama.cpp `llama-server` (Vulkan, pinned to the RTX 3050 Ti). Its 1.8 GB per-layer embedding table stays in RAM (`-ot per_layer_token_embd.weight=CPU`), so ~1.3 GB sits on the GPU: ~79 tok/s, answers start in 0.3–2.5 s. E4B doesn't fit in 4 GB. Fallback: Qwen3.5-2B. |
| Ears | **Whisper Hindi2Hinglish-Apex**: Whisper turbo fine-tuned by Oriserve on ~700 h of noisy Indian-accented Hindi. It writes Hindi as Hinglish ("music chalao"), and English stays English. Converted locally into `models/stt/`. Only English/Hindi are allowed. |
| Voice | Piper voices: `en_US-lessac` and `hi_IN-priyamvada`, ~0.3 s to speak a sentence on CPU |
| Hotkey | Super+Y → `~/.local/bin/youru-trigger` → socket → starts listening without the wake word |

**Tools (30).** These run straight away:
- web search, news, read a web page, weather
- time, system info (CPU, RAM, battery, GPU), list processes
- open an app or URL, volume, brightness, media play/pause/next, screenshot, notification, clipboard
- list, focus or switch windows and workspaces, lock the screen
- list, find and read files in your home folder

**Ask first** (say "yes"/"haan" or "no"/"nahi"):
- run a shell command
- write, move or trash a file (trash only, never permanent delete)
- close a window
- kill a program
- shutdown, reboot or suspend

Your recordings folder `data/` is write-protected from the assistant.

**Risk levels** (OmniPilot's classifier plus Youru's stricter floor, in `src/youru/tools/risk.py`):
- **Run straight away:** read-only commands (`ls`, `df`, `git status`…), Python that only computes, new files, keyboard shortcuts.
- **Ask first:** anything that changes or deletes, `sudo`, typing into a terminal, closing shortcuts (Super+Q…).
- **Always blocked:** `rm -rf /`, `dd` to a disk, fork bombs, and the like.

**More tools** (from [OmniPilot](https://github.com/xrchris7/OmniPilot), pinned at 0.1.0, wrapped in `tools/omnipilot.py`):
- `edit_file`, `copy_file`, `search_in_files` (grep), `http_request`, `download_file`, `list_drives`
- `run_python`: its own process, so variables persist between calls and a hang can't freeze Youru
- `press_keys` / `type_text`: Wayland keyboard control, needs `sudo pacman -S wtype`
- DigUp's `find_anything` / `find_code` / `open_file`: files by what's in them

**Hindi/Hinglish commands** are first rewritten into one English instruction ("Mere computer ko lock karo" →
"Lock the computer") on llama-server's second slot, so the small model picks the right tool. It still replies in Hindi.

`uv run youru log` shows what you said, the meaning, which tools ran, and the reply.

**Music and videos:**
- "\<song\> chalao" / "play \<song\>" plays the audio in a hidden mpv, with the top 5 results queued.
- "\<video\> dikhao" / "play \<video\> video" opens it playing on YouTube in the browser.
- "agla gaana", "pause", "music ruko" control it; "play the second one" picks from the last results.
- Uses `yt-dlp` (pip, inside the project).

**Everyday tools:**
- **Timers and reminders:** "5 minute ka timer lagao", "remind me at 7 pm to call mom", "kal subah 6 baje uthana". They're systemd timers, so they ring and speak even if Youru isn't running (`youru alert`).
- **Read the screen:** "what's on my screen", "is error ko padho", "translate this". Uses OCR, ~2–3 s, no GPU.
- **Deadlines:** "add exam deadline on 5 November", "is hafte kya due hai". These go into your `deadline-tracker`.
- **Notes and to-do:** "note likh lo: …", "mere todo dikhao". Plain markdown in `~/Documents/Youru/`.
- **"This":** for "summarize this" or "what is this", the focused window's title is attached so it knows what you mean.
- **Detail mode:** "explain in detail" / "poora samjhao" shows the full answer and speaks only the first few sentences.

**GLaDOS:** say "use GLaDOS voice" to switch English to the Portal GLaDOS voice (a fan-made Piper model, `models/tts/vits-piper-en_US-glados`, just as fast), and "normal voice" to switch back. Hindi keeps its voice; the choice is saved as `tts.english_voice`.

**Voices:** `uv run youru voices` writes Kokoro samples (cute English and Hindi girl voices) to `~/Music/youru-voice-samples/`. Pick one with `tts.engine: kokoro`, `tts.kokoro_voice`, `tts.kokoro_hindi_voice` and `tts.pitch` in config.yaml.

**Thinking:**
- Only "why / explain / compare / kyun / samjhao…" questions think first. Youru says "Hmm, let me think…" at once, and the answer starts in ~2.5 s.
- Thinking is capped at 48 tokens (`llm.think_budget`).
- Everything else answers in ~1 s without thinking.

**"I can't" → finds a way:**
- If its first sentence would be a refusal, that sentence is never spoken. It says "Let me find a way…" and looks up a recipe (`how_to`, `src/youru/recipes.yaml`).
- If no recipe fits, it searches the web, then runs the command. Read-only commands run; changes ask first.
- Example: "make Firefox my default browser" → `xdg-settings set …` after you say yes.

**Learning** (all in `~/.local/share/youru/`, plain files you can edit; `uv run youru learned` shows everything):
- **Facts:** "Mera naam Bharat hai, yaad rakhna", "remember that I live in …". It attaches them only when relevant, and names in them help speech-to-text.
- **Corrections:** "nahi, browser kholo" means the misheard phrase works next time.
- **Skills:** multi-step successes (screenshot → copy path) are saved as skills; commands found in solve mode are saved as recipes.
- **Nightly review** (`youru-learn.timer`, 23:50, only when plugged in): looks at the day's failures and proposes commands. They wait for `uv run youru learn --review`.

**Staying light:**
- The wake-word model only runs while someone is speaking (half the idle CPU).
- After 15 min without a command, Whisper is unloaded from the GPU (`assistant.sleep_after_min`) and reloads when the wake word fires.
- The LLM generates with 4 CPU threads.
- DigUp re-indexes every 2 h, only on mains power.
- The Super+Y hotkey client is a tiny C program (4 ms per press instead of 49 ms).

**LLM server:**

```fish
uv run youru llm status | start | stop
uv run youru llm bench       # compares 4B vs 2B speed with Whisper loaded, like real use
```

The server log is at `~/.cache/youru/llama-server.log`.

## Everyday use

```fish
uv run youru listen                 # live dashboard: speech meter, wake score, decisions
uv run youru listen --transcribe    # also prints what Whisper heard
uv run youru transcribe-test        # word error rate on your enrolled sentences (auto / en / hi modes)
uv run youru eval --all-models      # compare the 3 speaker models on your voice
```

## How it works

```
mic (16 kHz, 80 ms frames)
 ├─ VAD (Silero)          speech probability every 32 ms
 └─ wake word             fires only while speech is present; personal verifier on top
        │ wake
        ▼
 capture: 2.5 s pre-roll + command until 0.8 s of silence (or 4 s timeout / 15 s max)
        ▼
 voice lock: speech-only audio -> TitaNet embedding -> score vs your profile
        ├─ accept    -> owner callback (Phase 2: STT -> LLM -> actions)
        ├─ uncertain -> "please repeat"
        └─ reject
 every decision -> data/sessions/<date>/*.wav + sessions.jsonl
```

- Settings are in `config.yaml`. `youru eval` rewrites the thresholds in place.
- Every triggered utterance is saved in `data/sessions/`. That builds a personal dataset for checking (or fine-tuning) speech-to-text on your accent later.
- The profile slowly adapts: confidently accepted utterances are added, capped at 100, and the original enrollment is always kept.

## Code map

| File | Job |
|---|---|
| `src/youru/audio.py` | mic and file sources, ring buffer, levels |
| `src/youru/vad.py` | Silero VAD + start/end segmenter |
| `src/youru/wakeword.py` | openWakeWord (ONNX), VAD gating, refractory |
| `src/youru/speaker.py` | embeddings and the owner profile |
| `src/youru/pipeline.py` | state machine, speaker gate, session log, `build()` |
| `src/youru/record.py` | guided recording sessions and prompts |
| `src/youru/evaluate.py` | EER / FAR / FRR, false wakes per hour, report |
| `src/youru/stt.py` | faster-whisper (GPU, CPU fallback) |
| `src/youru/cli.py` | the `youru` command |

## Limits to know

- **Recordings can fool it.** The voice lock stops other people, but not a recording of your voice. Phase 2 will require a confirmation for risky actions (deleting files, running shell commands, payments).
- **"Youru" alone is short.** It sounds close to *your / you're*. The hard negatives, verifier and eval thresholds exist to fight that. If false wakes stay high, retrain on "Hey Youru".
- **VRAM is tight.** Whisper turbo uses ~1.5 GB of the 4 GB. A local LLM in Phase 2 will need CPU offload or model swapping.
