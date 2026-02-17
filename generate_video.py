#!/usr/bin/env python3
"""
generate_video.py — Local AI Video Generator (Image-to-Video)
Optimized for Apple Silicon M1 with 8GB RAM.
Uses Stable Video Diffusion (SVD-XT) via HuggingFace Diffusers.
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


# ─── Default Presets (from plan.md "Safety Preset") ──────────────────────────
SAFETY_PRESET = {
    "width": 448,
    "height": 256,
    "num_inference_steps": 20,
    "motion_bucket_id": 100,
    "fps": 6,
    "num_frames": 10,
    "noise_aug_strength": 0.02,
    "decode_chunk_size": 2,
}

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
OUTPUT_DIR = Path.home() / "Pictures" / "VideoAi" / "Output"


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_device():
    """Detect the best available compute device."""
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Metal (MPS) backend")
        return "mps"
    elif torch.cuda.is_available():
        print("🟢 Using NVIDIA CUDA backend")
        return "cuda"
    else:
        print("⚪ Using CPU (very slow)")
        return "cpu"


def check_memory():
    """Print available system memory as a sanity check."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        total_ram_gb = int(result.stdout.strip()) / (1024 ** 3)
        print(f"💾 System RAM: {total_ram_gb:.1f} GB")
        if total_ram_gb < 12:
            print("⚠️  Low RAM detected. Close all other apps for best results.")
    except Exception:
        pass


def prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load, validate, and resize input image."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img = img.resize((width, height), Image.LANCZOS)
    print(f"📷 Image loaded: {original_size} → resized to {img.size}")
    return img


def flush_memory():
    """Force garbage collection and clear MPS cache."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─── Pipeline Creation ───────────────────────────────────────────────────────

def create_pipeline(device: str = None):
    """
    Load the SVD pipeline with aggressive memory optimizations for 8GB RAM.
    Uses sequential CPU offloading: keeps model on CPU, moves only active
    component to MPS/GPU during each inference step.
    Returns the pipeline and the device string.
    """
    from diffusers import StableVideoDiffusionPipeline

    if device is None:
        device = get_device()

    print(f"\n⏳ Loading model: {MODEL_ID}")
    print("   (First run will download ~4GB — subsequent runs use cache)")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # ── Aggressive memory optimizations for 8GB ──
    # Sequential CPU offload: only 1 component on GPU at a time
    pipe.enable_sequential_cpu_offload(device=device)

    # Attention slicing: process attention in chunks
    pipe.enable_attention_slicing("max")

    # UNet forward chunking: break feedforward into smaller chunks
    if hasattr(pipe, "unet"):
        pipe.unet.enable_forward_chunking(chunk_size=1)

    print("✅ Model loaded with sequential CPU offload (8GB safe)\n")
    return pipe, device


# ─── Video Generation ────────────────────────────────────────────────────────

def generate_video(
    pipe,
    image: Image.Image,
    device: str,
    width: int = SAFETY_PRESET["width"],
    height: int = SAFETY_PRESET["height"],
    num_frames: int = SAFETY_PRESET["num_frames"],
    num_inference_steps: int = SAFETY_PRESET["num_inference_steps"],
    motion_bucket_id: int = SAFETY_PRESET["motion_bucket_id"],
    fps: int = SAFETY_PRESET["fps"],
    noise_aug_strength: float = SAFETY_PRESET["noise_aug_strength"],
    decode_chunk_size: int = SAFETY_PRESET["decode_chunk_size"],
    seed: int = None,
    progress_callback=None,
):
    """
    Generate a video from the input image using SVD.
    Returns a list of PIL frames.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print("🎬 Starting video generation...")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {num_frames} @ {fps} FPS")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Motion Intensity: {motion_bucket_id}")

    start_time = time.time()

    # Build callback for progress tracking
    def step_callback(pipe_instance, step, timestep, callback_kwargs):
        pct = int((step + 1) / num_inference_steps * 100)
        if progress_callback:
            progress_callback(pct, step + 1, num_inference_steps)
        else:
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r   Progress: [{bar}] {pct}%", end="", flush=True)
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


def save_video(frames, fps: int, output_path: str = None) -> str:
    """Export frames as an MP4 video file."""
    from diffusers.utils import export_to_video

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"video_{timestamp}.mp4")

    export_to_video(frames, output_path, fps=fps)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"💾 Video saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Generator — Image to Video (SVD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default safety preset (recommended for 8GB RAM)
  python generate_video.py photo.jpg

  # Custom parameters
  python generate_video.py photo.jpg --steps 25 --frames 20 --motion 80

  # Lowest memory usage
  python generate_video.py photo.jpg --width 448 --height 256 --frames 10 --steps 15
        """,
    )

    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--width", type=int, default=SAFETY_PRESET["width"],
                        help=f"Video width (default: {SAFETY_PRESET['width']})")
    parser.add_argument("--height", type=int, default=SAFETY_PRESET["height"],
                        help=f"Video height (default: {SAFETY_PRESET['height']})")
    parser.add_argument("--steps", type=int, default=SAFETY_PRESET["num_inference_steps"],
                        help=f"Sampling steps (default: {SAFETY_PRESET['num_inference_steps']})")
    parser.add_argument("--motion", type=int, default=SAFETY_PRESET["motion_bucket_id"],
                        help=f"Motion intensity 0-255 (default: {SAFETY_PRESET['motion_bucket_id']})")
    parser.add_argument("--fps", type=int, default=SAFETY_PRESET["fps"],
                        help=f"Output FPS (default: {SAFETY_PRESET['fps']})")
    parser.add_argument("--frames", type=int, default=SAFETY_PRESET["num_frames"],
                        help=f"Number of frames (default: {SAFETY_PRESET['num_frames']})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated)")

    args = parser.parse_args()

    # ── Preflight checks ──
    if not os.path.isfile(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    print("=" * 55)
    print("  🎬 AI Video Generator — Stable Video Diffusion")
    print("  🍎 Optimized for Apple Silicon (M1 8GB)")
    print("=" * 55)

    check_memory()

    # ── Load & generate ──
    image = prepare_image(args.image, args.width, args.height)
    pipe, device = create_pipeline()

    frames, fps = generate_video(
        pipe=pipe,
        image=image,
        device=device,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        motion_bucket_id=args.motion,
        fps=args.fps,
        seed=args.seed,
    )

    output_path = save_video(frames, fps, args.output)

    # ── Cleanup ──
    del pipe
    flush_memory()

    print(f"\n🎉 Done! Open your video:\n   open \"{output_path}\"")


if __name__ == "__main__":
    main()
