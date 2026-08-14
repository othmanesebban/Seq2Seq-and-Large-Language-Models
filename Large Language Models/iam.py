"""
File: inference_scene_captioning.py
Purpose: Détection automatique des scènes + captions par scène + résumé global (BART) + export SRT et vidéo sous-titrée.

Prérequis (installer avant d'exécuter ce script):
  pip install av moviepy torch transformers scenedetect
  # Pour le résumé global (optionnel):
  pip install transformers[sentencepiece]

Explication des blocs de code:
- Importations et configuration: modules nécessaires pour la vidéo, le ML et l'édition.
- Détection de scènes: utilise PySceneDetect pour trouver automatiquement les coupures.
- Config & Modèles: charge les modèles vidéo et texte une fois pour optimiser.
- Résumeur: prépare un modèle BART pour synthétiser un résumé global à partir des captions de scènes.
- Utils: fonctions utilitaires (extraction indices de frames).
- Captioning: génère un caption pour un segment vidéo en sélectionnant des frames.
- Résumé global: combine et résume toutes les captions de scènes pour raconter l’ensemble de la vidéo.
- Détection & montage: calcule les intervalles des scènes à partir des coupures.
- Overlays & rendu: génère une vidéo avec les captions incrustés (si activé).
- SRT helpers: écrit les sous-titres au format standard SRT.
- Pipeline principal: exécute l’ensemble (détection, captions par scène, résumé, exports JSON/SRT/vidéo).
- CLI: interface en ligne de commande permettant de lancer le script avec options.
"""

import os
import json
import time
import tempfile
import shutil
import argparse
from typing import List, Tuple

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from moviepy.editor import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# --- Détection de scènes ---
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception as exc:  # si dépendance manquante
    raise SystemExit(
        "PySceneDetect est requis. Installez-le avec: pip install scenedetect\n"
        f"Erreur originale: {exc}"
    )

# =====================
# Config & Modèles
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained(
    "Neleac/timesformer-gpt2-video-captioning"
).to(device)

# Résumeur (optionnel)
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
_summarizer = None

def _load_summarizer():
    """Charge le pipeline de résumé si disponible."""
    global _summarizer
    if _summarizer is not None:
        return _summarizer
    try:
        from transformers import pipeline
        _summarizer = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            device=0 if device == "cuda" else -1,
        )
    except Exception:
        _summarizer = None
    return _summarizer

# =====================
# Utils
# =====================

def _uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    """Retourne un ensemble d'indices uniformément répartis."""
    if n_total <= 0:
        return np.arange(0)
    n = min(max(int(n_pick), 1), n_total)
    return np.linspace(0, n_total - 1, num=n, endpoint=True).astype(np.int64)

# =====================
# Captioning vidéo/segment
# =====================

def caption_video_segment(video_path: str, min_len: int = 10, max_len: int = 20) -> str:
    """Produit un caption pour un segment vidéo donné."""
    container = av.open(video_path)
    try:
        seg_len = container.streams.video[0].frames or int(
            (float(container.streams.video[0].duration) / container.streams.video[0].time_base) if container.streams.video[0].duration else 0
        )
    except Exception:
        seg_len = 0

    clip_len = min(32, getattr(model.config.encoder, "num_frames", 16))
    indices = set(_uniform_indices(seg_len, clip_len))

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if not frames:
        raise RuntimeError("Impossible d'extraire des frames pour le captioning.")

    gen_kwargs = {"min_length": min_len, "max_length": max_len, "num_beams": 8}
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption.strip()

# =====================
# Résumé global à partir des captions de scènes
# =====================

def summarize_scene_captions(caps: List[str], max_words: int = 250) -> str:
    """Construit un résumé global cohérent à partir des captions de scènes."""
    text = " ".join([c.strip() for c in caps if c and c.strip()])
    if not text:
        return ""

    summarizer = _load_summarizer()
    if summarizer is None:
        return text

    words = text.split()
    chunks = []
    step = 300
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + step])
        chunks.append(chunk)

    partials = []
    for ch in chunks:
        try:
            out = summarizer(ch, max_length=180, min_length=60, do_sample=False)
            partials.append(out[0]["summary_text"])
        except Exception:
            partials.append(ch)

    merged = " ".join(partials)
    try:
        final = summarizer(merged, max_length=180, min_length=60, do_sample=False)
        return final[0]["summary_text"].strip()
    except Exception:
        return merged

# =====================
# Scène: détection & montage
# =====================

def detect_scenes(video_path: str, threshold: float = 30.0, min_scene_len: int = 15) -> List[Tuple[float, float]]:
    """Retourne les intervalles de scènes détectés dans la vidéo."""
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    spans: List[Tuple[float, float]] = []
    for start_time, end_time in scenes:
        s, e = start_time.get_seconds(), end_time.get_seconds()
        if e - s >= min_scene_len:
            spans.append((s, e))

    if not spans:
        with VideoFileClip(video_path) as v:
            spans = [(0.0, float(v.duration))]
    return spans

# =====================
# Overlays & rendu vidéo
# =====================

def overlay_caption_on_clip(clip: VideoFileClip, caption: str) -> CompositeVideoClip:
    """Superpose un texte de caption sur un clip vidéo."""
    txt = TextClip(
        caption,
        fontsize=36,
        color="white",
        method="caption",
        size=(clip.w - 80, None),
        bg_color="rgba(0,0,0,0.55)",
    ).set_duration(clip.duration).set_position(("center", "bottom"))
    return CompositeVideoClip([clip, txt])

# =====================
# SRT helpers
# =====================

def _fmt_ts(t: float) -> str:
    """Formate un timestamp float en format SRT hh:mm:ss,ms."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(spans: List[Tuple[float, float]], captions: List[str], srt_path: str) -> None:
    """Écrit un fichier SRT avec les captions par scène."""
    lines = []
    for i, ((s, e), cap) in enumerate(zip(spans, captions), start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(s)} --> {_fmt_ts(e)}")
        lines.append(cap)
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =====================
# Pipeline principal
# =====================

def process_video_with_scenes(
    video_path: str,
    out_dir: str = "result",
    threshold: float = 30.0,
    min_scene_len: int = 15,
    burn_subtitles: bool = True,
    crf: int = 18,
    preset: str = "medium",
    logger_mode: str = "none",  # "none" ou "bar"
) -> dict:
    """Pipeline complet: détecte scènes, génère captions, exporte JSON/SRT/vidéo."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    spans = detect_scenes(video_path, threshold=threshold, min_scene_len=min_scene_len)

    tmpdir = tempfile.mkdtemp(prefix="scenes_")
    per_scene_caps: List[str] = []
    scene_files: List[str] = []

    with VideoFileClip(video_path) as video:
        for idx, (s, e) in enumerate(spans):
            sub = video.subclip(s, e)
            scene_file = os.path.join(tmpdir, f"scene_{idx:03d}.mp4")
            sub.write_videofile(
                scene_file,
                codec="libx264",
                audio=False,
                preset=preset,
                ffmpeg_params=["-crf", str(crf)],
                verbose=False,
                logger=None if logger_mode == "none" else logger_mode,
            )
            scene_files.append(scene_file)
            cap = caption_video_segment(scene_file)
            per_scene_caps.append(cap)

    global_caption_brief = caption_video_segment(video_path)
    video_caption_overall = summarize_scene_captions(per_scene_caps) or global_caption_brief

    meta = {
        "video": video_path,
        "scenes": [
            {"index": i, "start": s, "end": e, "caption": c}
            for i, ((s, e), c) in enumerate(zip(spans, per_scene_caps))
        ],
        "global_caption_brief": global_caption_brief,
        "video_caption_overall": video_caption_overall,
    }
    json_path = os.path.join(out_dir, f"{base}_captions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    srt_path = os.path.join(out_dir, f"{base}.srt")
    write_srt(spans, per_scene_caps, srt_path)

    final_video_path = os.path.join(out_dir, f"{base}_with_scene_captions.mp4")
    if burn_subtitles:
        clips = []
        with VideoFileClip(video_path) as video:
            for cap, (s, e) in zip(per_scene_caps, spans):
                sub = video.subclip(s, e)
                clips.append(overlay_caption_on_clip(sub, cap))
            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(
                final_video_path,
                codec="libx264",
                audio_codec="aac",
                preset=preset,
                ffmpeg_params=["-crf", str(crf)],
                logger=None if logger_mode == "none" else logger_mode,
            )
    else:
        final_video_path = ""

    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "json": json_path,
        "srt": srt_path,
        "final_video": final_video_path,
        "global_caption_brief": global_caption_brief,
        "video_caption_overall": video_caption_overall,
        "scene_spans": spans,
        "scene_captions": per_scene_caps,
    }

# =====================
# CLI
# =====================

def parse_args():
    """Parse les arguments CLI pour personnaliser l'exécution."""
    p = argparse.ArgumentParser(description="Scene captioning + global summary")
    p.add_argument("inputs", nargs="+", help="Chemins des vidéos à traiter")
    p.add_argument("--out", default="result", help="Dossier de sortie")
    p.add_argument("--threshold", type=float, default=30.0, help="Seuil PySceneDetect")
    p.add_argument("--min-scene-len", type=int, default=10, help="Longueur min scène (s)")
    p.add_argument("--crf", type=int, default=20, help="Qualité x264 (plus petit = meilleure qualité)")
    p.add_argument("--preset", default="medium", help="Preset x264 (ultrafast..slow)")
    p.add_argument("--no-burn", action="store_true", help="Ne pas ré-encoder la vidéo (pas d'overlay)")
    p.add_argument("--progress", choices=["none", "bar"], default="none", help="Affichage progression MoviePy")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    for vp in args.inputs:
        base = os.path.splitext(os.path.basename(vp))[0]
        print(f"\n>>> Traitement: {vp}")
        start_time = time.time()
        outputs = process_video_with_scenes(
            vp,
            out_dir=args.out,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            burn_subtitles=not args.no_burn,
            crf=args.crf,
            preset=args.preset,
            logger_mode=args.progress,
        )
        elapsed = time.time() - start_time
        print(f"⏱ Temps de traitement pour {base}: {elapsed:.2f} s")
        print("JSON:", outputs["json"])
        print("SRT:", outputs["srt"])
        if outputs["final_video"]:
            print("Vidéo légendée:", outputs["final_video"])
        print("Caption global (brief):", outputs["global_caption_brief"])
        print("Résumé global (BART):", outputs["video_caption_overall"])

        # --- Sauvegarde d'un résumé texte du run ---
        log_path = os.path.join(args.out, f"{base}_summary.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Moviepy - Done !")
            if outputs["final_video"]:
                f.write(f"Moviepy - video ready {outputs['final_video']}")
            else:
                f.write("Moviepy - video ready (no-burn mode: aucune vidéo ré-encodée)")
            f.write(f"⏱ Temps de traitement pour {base}: {elapsed:.2f} s\n")
            f.write(f"JSON: {outputs['json']}\n")
            f.write(f"SRT: {outputs['srt']}\n")
            if outputs["final_video"]:
                f.write(f"Vidéo légendée: {outputs['final_video']}\n")
            f.write(f"Caption global (brief): {outputs['global_caption_brief']}\n")
            f.write(f"Résumé global (BART): {outputs['video_caption_overall']}\n")
        print(f"📄 Résumé sauvegardé dans {log_path}")
