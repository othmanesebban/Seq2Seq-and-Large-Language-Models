import os
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model, tokenizer, and processor
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
#model = VisionEncoderDecoderModel.from_pretrained("facebook/timesformer-base-finetuned-k600").to(device)
# Function to process a single video and generate caption
def process_video(video_path, output_path):
    # Load the video
    container = av.open(video_path)

    # Extract frames uniformly
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # Generate caption
    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(f"Generated Caption for {video_path}: {caption}")

    # Add the caption to the video
    add_caption_to_video(video_path, caption, output_path)

# Function to add the caption as text to the video
def add_caption_to_video(video_path, caption, output_path):
    # Load the video with moviepy
    video = VideoFileClip(video_path)

    # Create a text clip with the caption
    txt_clip = TextClip(caption, fontsize=24, color='white', size=(video.w, None), method='caption', bg_color='black')
    txt_clip = txt_clip.set_duration(video.duration).set_position(("center", "bottom"))

    # Overlay the text on the video
    result = CompositeVideoClip([video, txt_clip])

    # Export the video
    result.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Folder containing the videos
path = "video"
# List of input video filenames
input_videos = [os.path.join(path, "video31.mp4"), os.path.join(path, "video49.mp4"), os.path.join(path, "video123.mp4")]

# Process each video
for video_path in input_videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = f"result/result_{video_name}.mp4"
    process_video(video_path, output_video_path)
    print(f"Processed {video_path}, saved as {output_video_path}")
