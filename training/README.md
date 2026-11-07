# Training the "Youru" wake word — step by step

openWakeWord's **official** Colab notebook has been broken since late 2025 (upstream issue
[#296](https://github.com/dscripka/openWakeWord/issues/296)). Don't use it. This guide uses the community-fixed
notebook **[openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026)** (MIT licence), which
patches every known problem. Its exported model works directly with Youru. I checked this against the notebook on
2026-10-21.

The training mixes thousands of synthetic TTS clips with **your real recordings**, so the model learns your accent
and your laptop mic.

|  | Free Colab (T4 GPU) | Colab Pro (L4 GPU, $10/mo) |
|---|---|---|
| Total time | ~2.5–3 hours | ~75–90 min |
| Cost | free | about 1/100 of a month's credits |

Either works. Most of the time is downloads (~25 GB, on Google's servers, not your PC) and waiting.

---

## Part 1 — Record your clips (on your PC, ~20 min)

Open a terminal:

```fish
cd ~/Projects/youru
```

**Step 1. Mic check** (skip if you already did it)

```fish
uv run youru mic-check
```

**Step 2. Record "Youru" ~150 times**

```fish
uv run youru record wake
```

- Each batch shows an instruction (normal, fast, soft, lean back, fan on…). Press **Enter** to start the batch.
- When you see **"say it now"**, say **"Youru"** once, then pause. It saves the take and asks for the next one.
- **REDO** means it was too quiet or too short. Just say it again.
- Press `s` + Enter to skip a batch, or `q` + Enter to stop. Running it again later adds more takes.

**Step 3. Record sound-alike sentences** (these teach it what *not* to wake on)

```fish
uv run youru record confusables
```

Press **Enter**, read the sentence, then stop talking. Don't say "Youru" in these.

**Step 4. Make the upload pack**

```fish
uv run youru train-pack
```

This creates `~/Projects/youru/data/youru_train_pack.zip`, which you'll upload to Colab.

---

## Part 2 — Train in Google Colab (~1.5–3 h, mostly waiting)

**Step 1. Open the notebook**

Go to:
<https://colab.research.google.com/github/alfiedennen/openwakeword-colab-2026/blob/main/train_wakeword.ipynb>

Sign in with your Google account. If Colab warns *"This notebook was not authored by Google"*, click
**Run anyway** when it asks later.

**Step 2. Choose a GPU**

Go to **Runtime → Change runtime type**, choose **T4 GPU** (free) or **L4 GPU + High RAM** (Pro), and click **Save**.

**Step 3. Upload your pack**

Click the **📁 folder icon** in the left sidebar, then the **upload icon** (page with an up arrow). Choose
`youru_train_pack.zip` from `~/Projects/youru/data/`. Wait until it appears in the file list, under `/content`.

> Uploaded files disappear if the Colab session ends. If that happens, upload it again.

**Step 4. (Optional) Check how the TTS pronounces "youru"**

The synthetic voices should say it the way you do.

1. Scroll to the section **"4. Pre-flight"**.
2. Hover just below its code cell and click **+ Code** to add a new cell there.
3. Paste this in:

```python
import sys, glob
sys.path.insert(0, "/content/piper-sample-generator")
from generate_samples import generate_samples
from IPython.display import Audio, display

MODEL = "/content/piper-sample-generator/models/en_US-libritts_r-medium.pt"
for spelling in ["youru", "yooru", "yoo roo", "you roo", "yuru"]:
    out = "/content/pron/" + spelling.replace(" ", "_")
    generate_samples(text=[spelling], output_dir=out, model=MODEL, max_samples=3, batch_size=3)
    print(spelling)
    for f in sorted(glob.glob(out + "/*.wav")):
        display(Audio(f))
```

You'll run this cell in step 7, after the setup cells. Note which spelling sounds most like you.

**Step 5. Set the wake word name**

Scroll to **"10. Build training config"**. In its code cell, change the two lines at the top to:

```python
TARGET_PHRASE = ['youru']
MODEL_NAME    = 'youru'
```

**Step 6. Add the Youru settings and your real recordings**

Add **two** new cells (click **+ Code** under each section's code cell, as in step 4).

**Cell A**: add it right **after section 10's cell**. It applies `youru.yaml`: your sound-alike phrases and sample counts.

```python
!unzip -qo /content/youru_train_pack.zip -d /content/youru_pack
import yaml
cfg = yaml.safe_load(open("/content/my_model.yaml"))
cfg.update(yaml.safe_load(open("/content/youru_pack/youru.yaml")))
# cfg["target_phrase"] = ["yooru"]   # <- uncomment if step 4 preferred another spelling
yaml.dump(cfg, open("/content/my_model.yaml", "w"), sort_keys=False)
print(cfg["target_phrase"], cfg["n_samples"], len(cfg["custom_negative_phrases"]), "negative phrases")
```

**Cell B**: add it right **after section 12's cell ("Resample TTS clips")**, before section 13 ("Augment"). It
copies your real voice into the training data. This is the part that teaches the model your accent.

```python
import shutil, glob, os, yaml
cfg = yaml.safe_load(open("/content/my_model.yaml"))
# Each real take is copied 10x. Augmentation gives each copy different
# room echo, background noise and timing, so no two copies are the same.
for src, dest, times in [("real_positive_train", cfg["positive_clips_train_dir"], 10),
                         ("real_positive_test",  cfg["positive_clips_test_dir"],  1),
                         ("real_negative",       cfg["negative_clips_train_dir"], 5)]:
    files = glob.glob(f"/content/youru_pack/{src}/*.wav")
    for f in files:
        for k in range(times):
            shutil.copy(f, f"{dest}/real_{k}_{os.path.basename(f)}")
    print(f"{src}: {len(files)} clips x{times} -> {dest} (now {len(os.listdir(dest))} files)")
```

**Step 7. Run everything**

Click **Runtime → Run all**. Then leave it running:

- **Keep the browser tab open.** Free Colab disconnects if the tab is closed or idle for too long. Clicking the page
  now and then helps.
- Cell **"4. Pre-flight"** stops early with an error if anything is broken. If it passes, the rest will normally work.
- If the session disconnects, reconnect, upload the zip again (step 3) and **Run all** again. The notebook skips work
  it already finished.
- When **Cell B** runs, check that it printed numbers like `real_positive_train: 120 clips x10`. If it says
  `0 clips`, the zip wasn't uploaded.

**Step 8. Download the model**

The last cell downloads **`youru.onnx`** automatically through your browser. If it doesn't, open the 📁 panel, go to
`/content/youru_output/youru/`, right-click `youru.onnx`, and choose **Download**. (Ignore its message about
"android/app/..."; that's for the author's phone app.)

Near the end of its output, the notebook prints `positive test set: ... recall@0.5=...`. A value above **0.7** is a
good start.

---

## Part 3 — Install and tune it (on your PC, ~5 min + 45 min passive)

**Step 1. Put the model in place**

```fish
mv ~/Downloads/youru.onnx ~/Projects/youru/models/wakeword/
```

**Step 2. Tell Youru to use it**

Open `~/Projects/youru/config.yaml` and change the `model:` line under `wakeword:` to:

```yaml
  model: models/wakeword/youru.onnx
```

**Step 3. Record normal background** (for measuring false wakes)

```fish
uv run youru record background --minutes 45
```

Leave it running while you work, watch videos or talk normally, without saying "Youru". Ctrl+C stops early, and
finished minutes are kept.

**Step 4. Train your personal verifier** (takes seconds)

```fish
uv run youru train-verifier
```

**Step 5. Measure and pick the threshold**

```fish
uv run youru eval --wake
```

It prints a table of thresholds. Answer **y** to apply the suggested one. Targets: detection ≥ 90%, ≤ 0.5 false
wakes per hour.

**Step 6. Use it**

```fish
uv run youru listen --transcribe
```

Say **"Youru, music chalao"**. You should see **WAKE**, then **OWNER VERIFIED**, then what it heard.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| It wakes when you say "your" / "you're" | Run `youru record confusables` again (more sentences), then `youru train-verifier`. Also raise `wakeword.threshold` using the table in `data/eval_report.md`. |
| It misses you often | Run `youru record wake` again in the conditions where it misses (far away, fan on, soft voice), then retrain in Colab. Lowering `wakeword.threshold` a little is also safe, because the voice lock still blocks other people. |
| Recall in Colab was low (< 0.5) | In Cell A, set `cfg["n_samples"] = 5000` and retrain. |
| Still too many false wakes | In Cell A, add `cfg["max_negative_weight"] = 3000`, and add any phrases that wrongly woke it to `custom_negative_phrases` in `youru.yaml`. Clips of real false wakes are saved in `data/sessions/`. |
| Colab cell fails with an error | Copy the error text and send it to me. |
