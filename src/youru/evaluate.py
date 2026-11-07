"""Measure accuracy on *your* recordings and suggest thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import denoise, models
from .audio import FRAME, SAMPLE_RATE, load_wav, to_int16
from .config import DATA
from .record import BACKGROUND_DIR, CONFUSABLE_DIR, ENROLL_DIR, IMPOSTOR_DIR, WAKE_DIR
from .speaker import Embedder, OwnerProfile

REPORT = DATA / "eval_report.md"


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where it happens."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    best = (1.0, 0.0)
    gap = 2.0
    for t in thresholds:
        frr = float(np.mean(genuine < t))
        far = float(np.mean(impostor >= t))
        if abs(frr - far) < gap:
            gap, best = abs(frr - far), ((frr + far) / 2, float(t))
    return best


@dataclass
class SpeakerResult:
    model: str
    denoise: bool
    owner: np.ndarray
    impostor: np.ndarray
    eer: float | None = None
    accept: float = 0.0
    reject: float = 0.0
    frr: float = 0.0
    far: float | None = None
    notes: list[str] = field(default_factory=list)


def _speech(vad, path: Path, use_denoise: bool, noise) -> np.ndarray:
    x = vad.speech_only(load_wav(path))
    return denoise.reduce(x, noise) if use_denoise and len(x) else x


def speaker_eval(vad, model_names: list[str], min_speech_s: float = 1.5) -> list[SpeakerResult]:
    enroll = sorted(ENROLL_DIR.glob("*.wav"))
    if len(enroll) < 5:
        raise SystemExit("Need at least 5 enrollment clips. Run `youru enroll` first.")
    owner_extra = sorted(CONFUSABLE_DIR.glob("*.wav"))   # your own voice, not used for enrollment
    impostors = sorted(IMPOSTOR_DIR.glob("*/*.wav"))
    noise = denoise.load_noise_profile()
    results = []
    for name in model_names:
        emb = Embedder(models.ensure(name))
        for use_dn in ([False, True] if noise is not None else [False]):
            def embed_all(paths):
                out = []
                for p in paths:
                    x = _speech(vad, p, use_dn, noise)
                    if len(x) >= min_speech_s * SAMPLE_RATE:
                        out.append(emb.embed(x))
                return np.array(out)

            E = embed_all(enroll)
            # Leave-one-out for enrollment clips, full profile for everything else.
            owner = [OwnerProfile(emb.model_name, np.delete(E, i, 0)).score(E[i]) for i in range(len(E))]
            full = OwnerProfile(emb.model_name, E)
            owner += [full.score(e) for e in embed_all(owner_extra)]
            imp = np.array([full.score(e) for e in embed_all(impostors)]) if impostors else np.zeros(0)
            r = SpeakerResult(name, use_dn, np.array(owner), imp)
            if len(imp):
                r.eer, _ = eer(r.owner, imp)
                # FAR <= 1%: sit just above the impostor tail.
                tail = np.max(imp) if len(imp) < 100 else np.quantile(imp, 0.99)
                r.accept = round(float(tail) + 0.02, 3)
                r.far = float(np.mean(imp >= r.accept))
            else:
                r.accept = round(float(np.quantile(r.owner, 0.05)) - 0.05, 3)
                r.notes.append("no impostor clips: threshold is a guess from your own scores; "
                               "record others with `youru record impostor --name X`")
            r.reject = round(r.accept - 0.12, 3)
            r.frr = float(np.mean(r.owner < r.accept))
            results.append(r)
    return results


@dataclass
class WakeResult:
    model: str
    thresholds: list[float]
    detection: list[float]          # fraction of held-out takes detected
    bg_per_hour: list[float]        # false wakes per hour of background
    confusable_hits: list[int]      # false wakes on your confusable sentences
    n_pos: int
    bg_hours: float
    n_conf: int
    suggested: float


def _trace(wake, audio: np.ndarray) -> np.ndarray:
    wake.reset()
    x = np.concatenate([audio, np.zeros((-len(audio)) % FRAME, np.float32)])
    scores = np.array([max(wake.model.predict(to_int16(x[i:i + FRAME])).values())
                       for i in range(0, len(x), FRAME)])
    wake.reset()
    return scores


def _count_triggers(scores: np.ndarray, thr: float, refractory_frames: int) -> int:
    n, last = 0, -10**9
    for i in np.flatnonzero(scores >= thr):
        if i - last >= refractory_frames:
            n, last = n + 1, i
    return n


def wake_eval(wake) -> WakeResult:
    pos = sorted((WAKE_DIR / "test").glob("*.wav"))
    bg = sorted(BACKGROUND_DIR.glob("*.wav"))
    conf = sorted(CONFUSABLE_DIR.glob("*.wav"))
    if not pos:
        raise SystemExit("No held-out wake word takes. Run `youru record wake` first.")
    pos_max = np.array([wake.max_score(load_wav(p)) for p in pos])
    refr = int(wake.refractory * SAMPLE_RATE / FRAME)
    bg_traces = [_trace(wake, load_wav(p)) for p in bg]
    pad = np.zeros(SAMPLE_RATE // 2, np.float32)
    conf_traces = [_trace(wake, np.concatenate([pad, load_wav(p), pad])) for p in conf]
    bg_hours = sum(len(t) for t in bg_traces) * FRAME / SAMPLE_RATE / 3600
    thresholds = [round(t, 2) for t in np.arange(0.2, 0.96, 0.05)]
    det, bgh, ch = [], [], []
    for t in thresholds:
        det.append(float(np.mean(pos_max >= t)))
        n_bg = sum(_count_triggers(tr, t, refr) for tr in bg_traces)
        bgh.append(n_bg / bg_hours if bg_hours else float("nan"))
        ch.append(sum(_count_triggers(tr, t, refr) > 0 for tr in conf_traces))
    # Highest detection with <= 0.5 false wakes/hour and <= 5% confusable hits.
    ok = [i for i in range(len(thresholds))
          if (not bg_hours or bgh[i] <= 0.5) and (not conf or ch[i] <= 0.05 * len(conf))]
    best = max(ok, key=lambda i: (det[i], -thresholds[i])) if ok else len(thresholds) - 1
    return WakeResult(wake.name, thresholds, det, bgh, ch, len(pos), bg_hours, len(conf), thresholds[best])


def write_report(speaker: list[SpeakerResult] | None, wake: WakeResult | None) -> Path:
    lines = [f"# Youru evaluation report\n\n_{datetime.now():%Y-%m-%d %H:%M}_\n"]
    if speaker:
        lines += ["## Owner voice lock\n",
                  f"Owner clips scored: {len(speaker[0].owner)}, impostor clips: {len(speaker[0].impostor)}\n",
                  "| model | denoise | owner min / median | impostor max | EER | accept | FRR | FAR |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in speaker:
            imp_max = f"{r.impostor.max():.3f}" if len(r.impostor) else "-"
            lines.append(f"| {r.model} | {'on' if r.denoise else 'off'} | {r.owner.min():.3f} / "
                         f"{np.median(r.owner):.3f} | {imp_max} | "
                         f"{'-' if r.eer is None else f'{r.eer:.1%}'} | {r.accept:.3f} | {r.frr:.1%} | "
                         f"{'-' if r.far is None else f'{r.far:.1%}'} |")
        for n in {n for r in speaker for n in r.notes}:
            lines.append(f"\n> {n}")
        lines.append("\nTargets: FRR <= 5%, FAR <= 1%.\n")
    if wake:
        lines += [f"## Wake word ({wake.model})\n",
                  f"Held-out takes: {wake.n_pos}, background: {wake.bg_hours * 60:.0f} min, "
                  f"confusable sentences: {wake.n_conf}\n",
                  "| threshold | detection | false wakes / hour | confusable hits |", "|---|---|---|---|"]
        for t, d, b, c in zip(wake.thresholds, wake.detection, wake.bg_per_hour, wake.confusable_hits):
            mark = " **<- suggested**" if t == wake.suggested else ""
            lines.append(f"| {t:.2f} | {d:.0%} | {b:.2f} | {c} |{mark}")
        lines.append("\nTargets: detection >= 90%, <= 0.5 false wakes/hour (1 per 2 h).\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def best_speaker(results: list[SpeakerResult]) -> SpeakerResult:
    """Lowest EER (or lowest FRR without impostors); prefer no denoise on ties."""
    return min(results, key=lambda r: (r.eer if r.eer is not None else r.frr, r.frr, r.denoise))
