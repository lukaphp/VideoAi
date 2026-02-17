#!/usr/bin/env python3
"""
app.py — Flask Web Server for AI Video Generation
Supports text-to-video (CogVideoX) and image-to-video (SVD).
"""

import os
import sys
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime

# Prevent MPS OOM crashes
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
from PIL import Image
import io

# Lazy imports for generate_video module (requires torch)
_gen_module = None

def _get_gen():
    global _gen_module
    if _gen_module is None:
        import generate_video as gv
        _gen_module = gv
    return _gen_module


# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path.home() / "Pictures" / "VideoAi" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global State ────────────────────────────────────────────────────────────

pipelines = {}       # {"cogvideo": (pipe, device), "svd": (pipe, device)}
pipeline_loading = False
generation_state = {
    "status": "idle",        # idle | loading | generating | complete | error
    "progress": 0,
    "step": 0,
    "total_steps": 0,
    "message": "",
    "video_path": None,
    "video_id": None,
    "elapsed": 0,
}

state_lock = threading.Lock()


def update_state(**kwargs):
    with state_lock:
        generation_state.update(kwargs)


# ─── Pipeline Management ────────────────────────────────────────────────────

def ensure_pipeline(mode="cogvideo"):
    """Load pipeline lazily on first generation request."""
    global pipelines, pipeline_loading

    if mode in pipelines:
        return True

    if pipeline_loading:
        return False

    pipeline_loading = True
    model_labels = {
        "cogvideo": "CogVideoX-2B",
        "cogvideo_i2v": "CogVideoX-5B-I2V",
        "svd": "SVD"
    }
    model_name = model_labels.get(mode, mode)
    
    update_state(status="loading", message=f"Loading {model_name} model...", progress=0)

    try:
        gen = _get_gen()
        if mode == "cogvideo":
            pipe, device = gen.create_cogvideo_pipeline()
        elif mode == "cogvideo_i2v":
            pipe, device = gen.create_cogvideo_i2v_pipeline()
        else:
            pipe, device = gen.create_svd_pipeline()

        pipelines[mode] = (pipe, device)
        pipeline_loading = False
        update_state(status="idle", message=f"{model_name} loaded ✅")
        return True
    except Exception as e:
        pipeline_loading = False
        update_state(status="error", message=f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ─── Routes: Static ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─── Routes: API ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    try:
        device = _get_gen().get_device()
    except Exception:
        device = "unknown"

    return jsonify({
        "status": "ok",
        "model_loaded": len(pipelines) > 0,
        "model_loading": pipeline_loading,
        "device": device,
        "capabilities": {
            "text_to_video": device == "cuda",
            "image_to_video": True,
            "resolutions": ["480p", "720p", "1080p"],
            "fps_options": [6, 8, 25, 30],
            "durations": [2, 4, 6],
        },
    })


@app.route("/api/status", methods=["GET"])
def status():
    """SSE endpoint for real-time progress."""
    def event_stream():
        last_state = None
        while True:
            with state_lock:
                current = dict(generation_state)
            if current != last_state:
                yield f"data: {json.dumps(current)}\n\n"
                last_state = current.copy()
            if current["status"] in ("complete", "error", "idle"):
                if last_state and last_state["status"] in ("complete", "error"):
                    break
            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/generate", methods=["POST"])
def generate():
    """Start video generation from prompt and/or image."""
    if generation_state["status"] in ("loading", "generating"):
        return jsonify({"error": "Generation already in progress"}), 409

    # ── Parse parameters ──
    prompt = request.form.get("prompt", "").strip()
    resolution = request.form.get("resolution", "480p")
    duration = float(request.form.get("duration", 6))
    fps = int(request.form.get("fps", 8))
    steps = int(request.form.get("steps", 30))
    motion = int(request.form.get("motion", 100))
    guidance = float(request.form.get("guidance", 6.0))
    strength = float(request.form.get("strength", 0.8))
    seed = request.form.get("seed", None)

    if seed is not None and seed != "":
        seed = int(seed)
    else:
        # User requested NO RANDOMNESS ever.
        # We clamp the seed to a fixed number (e.g. 42) to ensure
        # the same image + prompt always yields the same result.
        seed = 42

    # ── Determine mode ──
    has_image = "image" in request.files and request.files["image"].filename != ""
    has_prompt = len(prompt) > 0

    if not has_prompt and not has_image:
        return jsonify({"error": "Provide a text prompt or upload an image"}), 400

    # Parse image if provided
    image = None
    if has_image:
        file = request.files["image"]
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

    video_id = str(uuid.uuid4())[:8]

    # ── Decide which pipeline to use ──
    # Prompt + Image → CogVideoX-5B-I2V (New!)
    # Prompt Only → CogVideoX-2B
    # Image Only → SVD
    use_cogvideo = has_prompt
    
    # If using CogVideoX, we need to handle image vs no-image inside logic
    # (Already handled in the updated logic block)

    def run_generation():
        try:
            gen = _get_gen()
            start_time = time.time()

            def progress_callback(pct, step, total):
                update_state(
                    status="generating",
                    progress=pct,
                    step=step,
                    total_steps=total,
                    message=f"Generating... Step {step}/{total}",
                )

            if use_cogvideo:
                if has_image:
                    # ── Image-to-Video with Prompt (CogVideoX-5B) ──
                    model_name = "CogVideoX-5B-I2V"
                    update_state(
                        status="loading",
                        message=f"Loading {model_name} (High Quality)...",
                        progress=0,
                        video_id=video_id,
                    )

                    if not ensure_pipeline("cogvideo_i2v"):
                        return

                    pipe, device = pipelines["cogvideo_i2v"]
                    
                    # Resize/Crop image to 600x480 (approx fits 12GB VRAM better than 720p)
                    # CogVideoX-5b native is 720x480 but 8-bit might need slightly less pixels for safety?
                    # Let's try native 720x480.
                    gen_w, gen_h = 720, 480
                    
                    # Prepare image
                    input_image = gen.prepare_image(request.files["image"], gen_w, gen_h)

                    # 8-bit inference often needs slightly fewer frames to be safe on 12GB?
                    # Let's stick to user request but cap at 49.
                    num_frames = min(int(duration * 8), 49)

                    update_state(
                        status="generating",
                        message=f"Generating with {model_name}...",
                        progress=0,
                    )

                    frames = gen.generate_video_from_image_prompt(
                        pipe=pipe,
                        prompt=prompt,
                        image=input_image,
                        device=device,
                        num_frames=num_frames,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        strength=strength, # Denoising strength
                        seed=seed,
                        progress_callback=progress_callback,
                    )
                    
                    output_fps = fps

                else:
                    # ── Text-to-Video (CogVideoX-2B) ──
                    update_state(
                        status="loading",
                        message="Loading CogVideoX-2B model...",
                        progress=0,
                        video_id=video_id,
                    )

                    if not ensure_pipeline("cogvideo"):
                        return

                    pipe, device = pipelines["cogvideo"]
                    res = gen.RESOLUTION_PRESETS[resolution]
                    gen_w, gen_h = res["gen"]
                    out_w, out_h = res["out"]

                    # CogVideoX generates at 8fps internally
                    num_frames = min(int(duration * 8), 49)

                    update_state(
                        status="generating",
                        message="Starting CogVideoX-2B generation...",
                        progress=0,
                    )

                    frames = gen.generate_video_from_prompt(
                        pipe=pipe,
                        prompt=prompt,
                        device=device,
                        num_frames=num_frames,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        gen_width=gen_w,
                        gen_height=gen_h,
                        seed=seed,
                        progress_callback=progress_callback,
                    )

                    # Upscale if needed
                    if (out_w, out_h) != (gen_w, gen_h):
                        update_state(message="Upscaling to target resolution...")
                        frames = gen.upscale_frames(frames, out_w, out_h)

                    output_fps = fps

            else:
                # ── Image-to-video with SVD ──
                update_state(
                    status="loading",
                    message="Loading SVD model...",
                    progress=0,
                    video_id=video_id,
                )

                if not ensure_pipeline("svd"):
                    return

                pipe, device = pipelines["svd"]

                # For SVD, resize image to generation size
                svd_w = int(request.form.get("width", gen.DEFAULTS["svd_width"]))
                svd_h = int(request.form.get("height", gen.DEFAULTS["svd_height"]))
                num_frames = int(request.form.get("frames", gen.DEFAULTS["svd_frames"]))
                resized_image = image.resize((svd_w, svd_h), Image.LANCZOS)

                update_state(
                    status="generating",
                    message="Starting SVD generation...",
                    progress=0,
                )

                frames, output_fps = gen.generate_video_from_image(
                    pipe=pipe,
                    image=resized_image,
                    device=device,
                    width=svd_w,
                    height=svd_h,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    motion_bucket_id=motion,
                    fps=fps,
                    seed=seed,
                    progress_callback=progress_callback,
                )

            elapsed = time.time() - start_time

            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(OUTPUT_DIR / f"video_{timestamp}_{video_id}.mp4")
            gen.save_video(frames, output_fps if not use_cogvideo else fps, output_path)

            update_state(
                status="complete",
                message="Video generated successfully! 🎉",
                progress=100,
                video_path=output_path,
                video_id=video_id,
                elapsed=round(elapsed, 1),
                seed=seed,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            update_state(
                status="error",
                message=f"Generation failed: {str(e)}",
                progress=0,
            )

        finally:
            _get_gen().flush_memory()

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    return jsonify({"status": "started", "video_id": video_id})


@app.route("/api/video/<video_id>", methods=["GET"])
def get_video(video_id):
    """Serve generated video file."""
    with state_lock:
        if generation_state.get("video_id") == video_id and generation_state.get("video_path"):
            path = generation_state["video_path"]
            if os.path.exists(path):
                return send_file(path, mimetype="video/mp4")
    return jsonify({"error": "Video not found"}), 404


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset generation state for a new run."""
    update_state(
        status="idle",
        progress=0,
        step=0,
        total_steps=0,
        message="",
        video_path=None,
        video_id=None,
        elapsed=0,
    )
    return jsonify({"status": "ok"})


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🎬 AI Video Generator — Web UI")
    print("  🌐 http://localhost:5001")
    print("=" * 55)
    try:
        _get_gen().check_memory()
    except Exception:
        print("💾 (Install torch to see memory info)")
    print("\n💡 Model will load on first generation request.\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
