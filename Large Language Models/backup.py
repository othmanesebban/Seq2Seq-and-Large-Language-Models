"""
File: inference_scene_captioning.py
Purpose: Détection automatique des scènes + caption par scène + caption global + export SRT et vidéo sous-titrée.

Prérequis (installer avant d'exécuter ce script):
  pip install av moviepy torch transformers scenedetect

Pourquoi ces choix:
- PySceneDetect pour détecter des changements visuels robustes.
- On charge les modèles (processor/tokenizer/model) une seule fois pour éviter la latence.
- On génère un SRT (interopérable) et une vidéo finale avec légendes gravées.
"""

import os
import json
import tempfile
import shutil
from typing import List, Tuple
import time
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
except Exception as exc:  # why: rappel explicite si la dépendance manque
    raise SystemExit(
        "PySceneDetect est requis. Installez-le avec: pip install scenedetect\n"
        f"Erreur originale: {exc}"
    )

# =====================
# Config & Modèles
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"

# why: garder le même pipeline que ton script existant
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained(
    "Neleac/timesformer-gpt2-video-captioning"
).to(device)

def _uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_total <= 0:
        return np.arange(0)
    n = min(max(int(n_pick), 1), n_total)
    return np.linspace(0, n_total - 1, num=n, endpoint=True).astype(np.int64)

# =====================
# Captioning vidéo/segment
# =====================

def caption_video_segment(video_path: str, min_len: int = 10, max_len: int = 20) -> str:
    """Génère un caption pour un fichier vidéo en échantillonnant uniformément des frames."""
    container = av.open(video_path)
    try:
        # why: nombre de frames peut être inconnu pour VFR; fallback via durée * fps
        seg_len = container.streams.video[0].frames or int(
            (float(container.streams.video[0].duration) / container.streams.video[0].time_base) if container.streams.video[0].duration else 0
        )
    except Exception:
        seg_len = 0

    clip_len = getattr(model.config.encoder, "num_frames", 16)
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
# Scène: détection & montage
# =====================

def detect_scenes(video_path: str, threshold: float = 30.0, min_scene_len: int = 15) -> List[Tuple[float, float]]:
    """Retourne une liste [(start_sec, end_sec), ...] pour chaque scène.
    min_scene_len en secondes pour filtrer les scènes trop courtes.
    """
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
        #if e - s >= min_scene_len: avec filtrage
        spans.append((s, e))

    if not spans:  # why: fallback pour ne pas renvoyer vide
        with VideoFileClip(video_path) as v:
            spans = [(0.0, float(v.duration))]
    return spans

# =====================
# Overlays & rendu vidéo
# =====================

def overlay_caption_on_clip(clip: VideoFileClip, caption: str) -> CompositeVideoClip:
    # why: fond semi-opaque pour lisibilité
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
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(spans: List[Tuple[float, float]], captions: List[str], srt_path: str) -> None:
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
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    # 1) Détection des scènes
    spans = detect_scenes(video_path, threshold=threshold, min_scene_len=min_scene_len)

    # 2) Caption par scène
    tmpdir = tempfile.mkdtemp(prefix="scenes_")
    per_scene_caps: List[str] = []
    scene_files: List[str] = []

    with VideoFileClip(video_path) as video:
        for idx, (s, e) in enumerate(spans):
            sub = video.subclip(s, e)
            scene_file = os.path.join(tmpdir, f"scene_{idx:03d}.mp4")
            # why: audio False -> accélère l'encodage pour le captioning
            sub.write_videofile(
                scene_file,
                codec="libx264",
                audio=False,
                preset=preset,
                ffmpeg_params=["-crf", str(crf)],
                verbose=False,
                logger=None,
            )
            scene_files.append(scene_file)
            cap = caption_video_segment(scene_file)
            per_scene_caps.append(cap)

    # 3) Caption global (sur la vidéo entière)
    global_caption = caption_video_segment(video_path)

    # 4) Exports texte
    meta = {
        "video": video_path,
        "scenes": [
            {"index": i, "start": s, "end": e, "caption": c}
            for i, ((s, e), c) in enumerate(zip(spans, per_scene_caps))
        ],
        "global_caption": global_caption,
    }
    json_path = os.path.join(out_dir, f"{base}_captions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    srt_path = os.path.join(out_dir, f"{base}.srt")
    write_srt(spans, per_scene_caps, srt_path)

    # 5) Vidéo finale avec sous-titres par scène
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
            )
    else:
        final_video_path = ""

    # Nettoyage
    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "json": json_path,
        "srt": srt_path,
        "final_video": final_video_path,
        "global_caption": global_caption,
        "scene_spans": spans,
        "scene_captions": per_scene_caps,
    }

# =====================
# Exemple d'utilisation (batch)
# =====================
if __name__ == "__main__":
    path = "video"
    input_videos = [
        os.path.join(path, "test3.mp4"),
        # os.path.join(path, "video49.mp4"),
        # os.path.join(path, "video123.mp4"),
    ]

    os.makedirs("result", exist_ok=True)

    for vp in input_videos:
        base = os.path.splitext(os.path.basename(vp))[0]
        print(f"\n>>> Traitement: {vp}")
        start_time = time.time()

        outputs = process_video_with_scenes(
            vp,
            out_dir="result",
            threshold=30.0,      # ↑ pour moins de scènes, ↓ pour plus
            min_scene_len=10,    # en secondes
            burn_subtitles=True,
            crf=20,
            preset="medium",
        )

        elapsed = time.time() - start_time

        log_text = (
            f"\n>>> Traitement: {vp}\n"
            f"⏱ Temps de traitement pour {base}: {elapsed:.2f} secondes\n"
            f"Captions exportés: {outputs['json']}\n"
            f"SRT exporté: {outputs['srt']}\n"
            f"Vidéo légendée: {outputs['final_video'] if outputs['final_video'] else 'Non générée'}\n"
            f"Caption global: {outputs['global_caption']}\n"
        )

        print(log_text)

        # --- Enregistrement dans un fichier résumé ---
        log_path = os.path.join("result", f"{base}_summary.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_text)
        print(f"📄 Résumé sauvegardé dans {log_path}")

