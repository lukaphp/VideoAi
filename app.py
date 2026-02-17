#!/usr/bin/env python3
"""
app.py — Flask Web Server for AI Video Generation
Serves a premium web UI and handles video generation requests.
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

# Default preset (duplicated for when torch isn't installed yet)
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

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path.home() / "Pictures" / "VideoAi" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global State ────────────────────────────────────────────────────────────

pipeline = None
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

def ensure_pipeline():
    """Load pipeline lazily on first generation request."""
    global pipeline, pipeline_loading

    if pipeline is not None:
        return True

    if pipeline_loading:
        return False

    pipeline_loading = True
    update_state(status="loading", message="Loading SVD model (~60s first time)...", progress=0)

    try:
        pipe, device = _get_gen().create_pipeline()
        pipeline = (pipe, device)
        pipeline_loading = False
        update_state(status="idle", message="Model loaded ✅")
        return True
    except Exception as e:
        pipeline_loading = False
        update_state(status="error", message=f"Failed to load model: {str(e)}")
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
        "model_loaded": pipeline is not None,
        "model_loading": pipeline_loading,
        "device": device,
        "preset": SAFETY_PRESET,
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
    """Start video generation from uploaded image."""
    if generation_state["status"] in ("loading", "generating"):
        return jsonify({"error": "Generation already in progress"}), 409

    # ── Parse uploaded image ──
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ── Parse parameters ──
    params = {
        "width": int(request.form.get("width", SAFETY_PRESET["width"])),
        "height": int(request.form.get("height", SAFETY_PRESET["height"])),
        "num_inference_steps": int(request.form.get("steps", SAFETY_PRESET["num_inference_steps"])),
        "motion_bucket_id": int(request.form.get("motion", SAFETY_PRESET["motion_bucket_id"])),
        "fps": int(request.form.get("fps", SAFETY_PRESET["fps"])),
        "num_frames": int(request.form.get("frames", SAFETY_PRESET["num_frames"])),
        "seed": request.form.get("seed", None),
    }
    if params["seed"] is not None and params["seed"] != "":
        params["seed"] = int(params["seed"])
    else:
        params["seed"] = None

    # ── Save uploaded image ──
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((params["width"], params["height"]), Image.LANCZOS)

    video_id = str(uuid.uuid4())[:8]

    # ── Start generation in background ──
    def run_generation():
        try:
            update_state(
                status="loading",
                message="Loading model...",
                progress=0,
                video_id=video_id,
            )

            if not ensure_pipeline():
                return

            pipe, device = pipeline

            def progress_callback(pct, step, total):
                update_state(
                    status="generating",
                    progress=pct,
                    step=step,
                    total_steps=total,
                    message=f"Generating frame data... Step {step}/{total}",
                )

            update_state(
                status="generating",
                message="Starting generation...",
                progress=0,
            )

            start_time = time.time()

            frames, fps = _get_gen().generate_video(
                pipe=pipe,
                image=image,
                device=device,
                width=params["width"],
                height=params["height"],
                num_frames=params["num_frames"],
                num_inference_steps=params["num_inference_steps"],
                motion_bucket_id=params["motion_bucket_id"],
                fps=params["fps"],
                seed=params["seed"],
                progress_callback=progress_callback,
            )

            elapsed = time.time() - start_time

            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(OUTPUT_DIR / f"video_{timestamp}_{video_id}.mp4")
            _get_gen().save_video(frames, fps, output_path)

            update_state(
                status="complete",
                message="Video generated successfully! 🎉",
                progress=100,
                video_path=output_path,
                video_id=video_id,
                elapsed=round(elapsed, 1),
            )

        except Exception as e:
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
