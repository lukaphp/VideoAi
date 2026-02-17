#!/usr/bin/env python3
"""
generate_video.py — AI Video Generator (Text-to-Video + Image-to-Video)
Supports:
  - CogVideoX-2B: text-to-video with prompt support (CUDA 12GB+)
  - SVD-XT: image-to-video fallback (MPS/CUDA)
  - Post-generation upscaling to 720p / 1080p
"""

import os
import sys
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime

# Prevent MPS OOM crashes on low-memory systems
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from PIL import Image
from tqdm import tqdm


# ─── Models & Config ─────────────────────────────────────────────────────────

MODEL_COGVIDEO = "THUDM/CogVideoX-2b"
MODEL_SVD = "stabilityai/stable-video-diffusion-img2vid-xt"
OUTPUT_DIR = Path.home() / "Pictures" / "VideoAi" / "Output"

# Resolution presets: (generation_w, generation_h, output_w, output_h)
RESOLUTION_PRESETS = {
    "480p":  {"gen": (720, 480),   "out": (720, 480)},
    "720p":  {"gen": (720, 480),   "out": (1280, 720)},
    "1080p": {"gen": (720, 480),   "out": (1920, 1080)},
}

# Default parameters
DEFAULTS = {
    "resolution": "480p",
    "num_inference_steps": 30,
    "fps": 8,
    "duration": 6,        # seconds
    "guidance_scale": 6.0,
    # SVD fallback
    "svd_width": 448,
    "svd_height": 256,
    "svd_frames": 10,
    "svd_fps": 6,
    "motion_bucket_id": 100,
    "noise_aug_strength": 0.02,
    "decode_chunk_size": 2,
}


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_device():
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        print("🟢 Using NVIDIA CUDA backend")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple Metal (MPS) backend")
        return "mps"
    else:
        print("⚪ Using CPU (very slow)")
        return "cpu"


def check_memory():
    """Print available system memory as a sanity check."""
    try:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            name = torch.cuda.get_device_name(0)
            print(f"💾 GPU: {name} — {vram:.1f} GB VRAM")
        else:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            total_ram_gb = int(result.stdout.strip()) / (1024 ** 3)
            print(f"💾 System RAM: {total_ram_gb:.1f} GB")
    except Exception:
        print("💾 (Install torch to see memory info)")


def prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load, validate, and resize input image."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img = img.resize((width, height), Image.LANCZOS)
    print(f"📷 Image loaded: {original_size} → resized to {img.size}")
    return img


def flush_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def upscale_frames(frames, target_w, target_h):
    """Upscale a list of PIL/numpy frames to target resolution using Lanczos."""
    import numpy as np
    upscaled = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            pil_frame = frame
        elif hasattr(frame, "numpy"):
            arr = frame.numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil_frame = Image.fromarray(arr)
        else:
            arr = np.array(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil_frame = Image.fromarray(arr)

        if pil_frame.size != (target_w, target_h):
            pil_frame = pil_frame.resize((target_w, target_h), Image.LANCZOS)
        upscaled.append(pil_frame)
    print(f"🔍 Upscaled {len(upscaled)} frames to {target_w}x{target_h}")
    return upscaled


# ─── CogVideoX Pipeline (Text-to-Video) ─────────────────────────────────────

def create_cogvideo_pipeline(device: str = None):
    """Load CogVideoX-2B pipeline optimized for 12GB VRAM."""
    from diffusers import CogVideoXPipeline

    if device is None:
        device = get_device()

    print(f"\n⏳ Loading model: {MODEL_COGVIDEO}")
    print("   (First run will download ~9GB — subsequent runs use cache)")

    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_COGVIDEO,
        torch_dtype=torch.float16,
    )

    # Memory optimizations for 12GB
    pipe.enable_sequential_cpu_offload(device=device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("✅ CogVideoX-2B loaded (text-to-video ready)\n")
    return pipe, device


def generate_video_from_prompt(
    pipe,
    prompt: str,
    device: str,
    num_frames: int = 49,
    num_inference_steps: int = DEFAULTS["num_inference_steps"],
    guidance_scale: float = DEFAULTS["guidance_scale"],
    gen_width: int = 720,
    gen_height: int = 480,
    seed: int = None,
    progress_callback=None,
):
    """Generate video from text prompt using CogVideoX."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print("🎬 Starting CogVideoX text-to-video generation...")
    print(f"   Prompt: {prompt[:80]}...")
    print(f"   Resolution: {gen_width}x{gen_height}")
    print(f"   Frames: {num_frames}")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Guidance: {guidance_scale}")

    start_time = time.time()

    # Build callback for progress tracking
    def step_callback(pipe_instance, step, timestep, callback_kwargs):
        pct = int((step + 1) / num_inference_steps * 100)
        if progress_callback:
            progress_callback(pct, step + 1, num_inference_steps)
        return callback_kwargs

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=gen_width,
            height=gen_height,
            generator=generator,
            callback_on_step_end=step_callback,
        )

    elapsed = time.time() - start_time
    print(f"\n\n✅ Generation complete in {elapsed:.1f} seconds")

    flush_memory()
    return output.frames[0]


# ─── SVD Pipeline (Image-to-Video, fallback) ────────────────────────────────

def create_svd_pipeline(device: str = None):
    """Load the SVD pipeline with memory optimizations."""
    from diffusers import StableVideoDiffusionPipeline

    if device is None:
        device = get_device()

    print(f"\n⏳ Loading model: {MODEL_SVD}")
    print("   (First run will download ~4GB — subsequent runs use cache)")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_SVD,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    pipe.enable_sequential_cpu_offload(device=device)
    pipe.enable_attention_slicing("max")
    if hasattr(pipe, "unet"):
        pipe.unet.enable_forward_chunking(chunk_size=1)

    print("✅ SVD loaded (image-to-video ready)\n")
    return pipe, device


def generate_video_from_image(
    pipe,
    image: Image.Image,
    device: str,
    width: int = DEFAULTS["svd_width"],
    height: int = DEFAULTS["svd_height"],
    num_frames: int = DEFAULTS["svd_frames"],
    num_inference_steps: int = 20,
    motion_bucket_id: int = DEFAULTS["motion_bucket_id"],
    fps: int = DEFAULTS["svd_fps"],
    noise_aug_strength: float = DEFAULTS["noise_aug_strength"],
    decode_chunk_size: int = DEFAULTS["decode_chunk_size"],
    seed: int = None,
    progress_callback=None,
):
    """Generate a video from the input image using SVD."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print("🎬 Starting SVD image-to-video generation...")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {num_frames} @ {fps} FPS")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Motion Intensity: {motion_bucket_id}")

    start_time = time.time()

    def step_callback(pipe_instance, step, timestep, callback_kwargs):
        pct = int((step + 1) / num_inference_steps * 100)
        if progress_callback:
            progress_callback(pct, step + 1, num_inference_steps)
        return callback_kwargs

    with torch.no_grad():
        output = pipe(
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
            callback_on_step_end=step_callback,
        )

    elapsed = time.time() - start_time
    print(f"\n\n✅ Generation complete in {elapsed:.1f} seconds")

    flush_memory()
    return output.frames[0], fps


# ─── Unified Video Save ─────────────────────────────────────────────────────

def save_video(frames, fps: int, output_path: str = None) -> str:
    """Export frames as an MP4 video file (H.264 for browser compatibility)."""
    import imageio
    import numpy as np

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"video_{timestamp}.mp4")

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8,
                                 output_params=["-pix_fmt", "yuv420p"])
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        elif hasattr(frame, "numpy"):
            frame = frame.numpy()
        if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        writer.append_data(np.array(frame))
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"💾 Video saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Generator — Text-to-Video & Image-to-Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-video with prompt (requires CUDA GPU)
  python generate_video.py --prompt "A cat walking in a garden"

  # Image-to-video (works on Mac M1 too)
  python generate_video.py --image photo.jpg

  # Text-to-video at 720p
  python generate_video.py --prompt "Ocean waves at sunset" --resolution 720p --fps 25
        """,
    )

    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Text prompt describing the video to generate")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to input image (for image-to-video)")
    parser.add_argument("--resolution", "-r", type=str, default="480p",
                        choices=["480p", "720p", "1080p"],
                        help="Output resolution (default: 480p)")
    parser.add_argument("--duration", type=float, default=6,
                        help="Video duration in seconds (default: 6)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Output FPS (default: 8)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Sampling steps (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated)")

    args = parser.parse_args()

    if not args.prompt and not args.image:
        parser.error("Provide --prompt (text-to-video) or --image (image-to-video)")

    print("=" * 55)
    print("  🎬 AI Video Generator")
    print("=" * 55)

    check_memory()

    if args.prompt:
        # ── Text-to-video with CogVideoX ──
        res = RESOLUTION_PRESETS[args.resolution]
        gen_w, gen_h = res["gen"]
        out_w, out_h = res["out"]

        # CogVideoX generates at 8 fps internally, num_frames = duration * 8
        # but it supports specific frame counts: 1-49
        num_frames = min(int(args.duration * 8), 49)

        pipe, device = create_cogvideo_pipeline()
        frames = generate_video_from_prompt(
            pipe=pipe,
            prompt=args.prompt,
            device=device,
            num_frames=num_frames,
            num_inference_steps=args.steps,
            gen_width=gen_w,
            gen_height=gen_h,
            seed=args.seed,
        )

        # Upscale if needed
        if (out_w, out_h) != (gen_w, gen_h):
            frames = upscale_frames(frames, out_w, out_h)

        output_path = save_video(frames, args.fps, args.output)
        del pipe
    else:
        # ── Image-to-video with SVD ──
        image = prepare_image(args.image, DEFAULTS["svd_width"], DEFAULTS["svd_height"])
        pipe, device = create_svd_pipeline()
        frames, fps = generate_video_from_image(
            pipe=pipe,
            image=image,
            device=device,
            seed=args.seed,
        )
        output_path = save_video(frames, fps, args.output)
        del pipe

    flush_memory()
    print(f"\n🎉 Done! Open your video:\n   open \"{output_path}\"")


if __name__ == "__main__":
    main()
