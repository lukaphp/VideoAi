# 🎬 VideoAi — Local AI Video Generator

Generate videos from images using AI, **100% locally** on your Mac. Powered by [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) with Apple Metal (MPS) acceleration.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0+-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black?logo=apple)

---

## ✨ Features

- 🖼️ **Image-to-Video** — Upload any image, get an animated video
- 🌐 **Web UI** — Premium dark glassmorphism interface with drag & drop
- 🍎 **Apple Silicon optimized** — Runs on M1/M2/M3 with MPS backend
- 🔒 **100% local & private** — No data leaves your machine
- ⚡ **Safety Presets** — One-click optimal settings for 8GB RAM
- 📊 **Real-time progress** — SSE-powered live progress bar
- 🎞️ **Video preview & download** — Built-in player with MP4 export

---

## 🚀 Quick Start

### Prerequisites
- macOS 12.6+ (13.0+ recommended)
- Python 3.10+
- Apple Silicon Mac (M1/M2/M3)

### Installation
```bash
git clone https://github.com/lukaphp/VideoAi.git
cd VideoAi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision diffusers transformers accelerate opencv-python imageio imageio-ffmpeg
```

### Run Web UI
```bash
source venv/bin/activate
python app.py
```
Open **http://localhost:5001** in your browser.

### Run CLI
```bash
# Default preset (recommended for 8GB RAM)
python generate_video.py photo.jpg

# Custom parameters
python generate_video.py photo.jpg --steps 25 --frames 14 --motion 80

# Minimal memory usage
python generate_video.py photo.jpg --width 256 --height 256 --frames 8 --steps 15
```

---

## ⚙️ Parameters

| Parameter | Default | Range | Description |
|:---|:---|:---|:---|
| Width | 448 | 256-1024 | Video width (multiples of 64) |
| Height | 256 | 256-1024 | Video height (multiples of 64) |
| Steps | 20 | 5-50 | Sampling steps (more = better quality) |
| Motion | 100 | 1-255 | Motion intensity |
| FPS | 6 | 2-30 | Frames per second |
| Frames | 10 | 4-25 | Number of frames |
| Seed | random | 0-999999999 | Reproducibility seed |

---

## 🏗️ Architecture

```
Browser ⟶ Flask (app.py) ⟶ generate_video.py ⟶ SVD Pipeline (MPS)
              │                                         │
              └─ SSE progress ◄────────────────────────┘
```

```
VideoAi/
├── app.py                 # Flask server + REST API
├── generate_video.py      # SVD pipeline + CLI
├── requirements.txt       # Python dependencies
├── plan.md                # Setup guide (Italian)
├── guida_cloud_gpu.md     # Cloud GPU guide (Italian)
├── static/
│   ├── index.html         # Web UI
│   ├── style.css          # Dark glassmorphism theme
│   └── app.js             # Frontend logic
└── README.md
```

---

## 💾 Memory Optimizations

This project uses aggressive optimizations to run on 8GB RAM:

- **Sequential CPU Offload** — Only 1 model component on GPU at a time
- **Attention Slicing (max)** — Chunked attention computation
- **UNet Forward Chunking** — Feedforward layers processed in chunks
- **Float16 precision** — Half the memory of float32
- **MPS High Watermark disabled** — Prevents premature OOM errors

---

## ⚠️ Known Limitations

- **No text prompts** — SVD is image-to-video only (no text conditioning)
- **Slow on 8GB** — ~10 min per video due to CPU offloading
- **Short videos** — Max ~14 frames (2.3s at 6fps) on 8GB

---

## 📄 License

MIT License — free for personal and commercial use.
