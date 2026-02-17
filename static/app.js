/**
 * app.js — VideoAi Frontend Logic
 * Handles image upload, parameter controls, API communication, and video playback.
 */

// ─── DOM Elements ────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const uploadZone = $("#uploadZone");
const uploadContent = $("#uploadContent");
const imagePreview = $("#imagePreview");
const btnRemove = $("#btnRemove");
const fileInput = $("#fileInput");
const btnGenerate = $("#btnGenerate");
const btnPreset = $("#btnPreset");
const btnDownload = $("#btnDownload");
const btnNewVideo = $("#btnNewVideo");
const progressSection = $("#progressSection");
const progressFill = $("#progressFill");
const progressMessage = $("#progressMessage");
const progressPct = $("#progressPct");
const progressTimer = $("#progressTimer");
const videoOutput = $("#videoOutput");
const videoPlayer = $("#videoPlayer");
const statusDot = $("#statusDot");
const statusLabel = $("#statusLabel");

// Parameter controls
const controls = {
    width: { el: $("#width"), display: $("#widthValue") },
    height: { el: $("#height"), display: $("#heightValue") },
    steps: { el: $("#steps"), display: $("#stepsValue") },
    motion: { el: $("#motion"), display: $("#motionValue") },
    fps: { el: $("#fps"), display: $("#fpsValue") },
    frames: { el: $("#frames"), display: $("#framesValue") },
};
const seedInput = $("#seed");
const seedLabel = $("#seedLabel");

// State
let uploadedFile = null;
let generationTimer = null;
let generationStartTime = null;
let eventSource = null;

// ─── Safety Preset ───────────────────────────────────────────────────────────

const SAFETY_PRESET = {
    width: 448,
    height: 256,
    steps: 20,
    motion: 100,
    fps: 6,
    frames: 10,
};

function applyPreset() {
    Object.entries(SAFETY_PRESET).forEach(([key, value]) => {
        if (controls[key]) {
            controls[key].el.value = value;
            controls[key].display.textContent = value;
        }
    });
    seedInput.value = "";
    seedLabel.textContent = "Random";

    // Flash animation
    btnPreset.style.transform = "scale(0.95)";
    setTimeout(() => { btnPreset.style.transform = ""; }, 150);
}

// ─── Slider Binding ──────────────────────────────────────────────────────────

Object.entries(controls).forEach(([key, ctrl]) => {
    ctrl.el.addEventListener("input", () => {
        ctrl.display.textContent = ctrl.el.value;
    });
});

seedInput.addEventListener("input", () => {
    seedLabel.textContent = seedInput.value ? seedInput.value : "Random";
});

btnPreset.addEventListener("click", applyPreset);

// ─── Image Upload ────────────────────────────────────────────────────────────

function handleFile(file) {
    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
        alert("Formato non supportato. Usa JPG, PNG o WebP.");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert("Immagine troppo grande. Max 10MB.");
        return;
    }

    uploadedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove("hidden");
        btnRemove.classList.remove("hidden");
        uploadContent.classList.add("hidden");
        uploadZone.classList.add("has-image");
        btnGenerate.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    uploadedFile = null;
    imagePreview.src = "";
    imagePreview.classList.add("hidden");
    btnRemove.classList.add("hidden");
    uploadContent.classList.remove("hidden");
    uploadZone.classList.remove("has-image");
    btnGenerate.disabled = true;
    fileInput.value = "";
}

// Click to upload
uploadZone.addEventListener("click", (e) => {
    if (e.target === btnRemove || e.target.closest(".btn-remove")) return;
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
});

// Drag & Drop
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    removeImage();
});

// ─── Generation ──────────────────────────────────────────────────────────────

async function startGeneration() {
    if (!uploadedFile) return;

    // Hide previous video
    videoOutput.classList.add("hidden");

    // Build FormData
    const formData = new FormData();
    formData.append("image", uploadedFile);
    formData.append("width", controls.width.el.value);
    formData.append("height", controls.height.el.value);
    formData.append("steps", controls.steps.el.value);
    formData.append("motion", controls.motion.el.value);
    formData.append("fps", controls.fps.el.value);
    formData.append("frames", controls.frames.el.value);
    if (seedInput.value) {
        formData.append("seed", seedInput.value);
    }

    // UI: Show progress
    progressSection.classList.remove("hidden");
    progressFill.style.width = "0%";
    progressFill.classList.add("active");
    progressMessage.textContent = "Invio immagine...";
    progressPct.textContent = "0%";
    btnGenerate.disabled = true;
    btnGenerate.querySelector(".btn-generate-text").textContent = "Generazione in corso...";

    setStatus("loading", "Generazione...");

    // Start timer
    generationStartTime = Date.now();
    generationTimer = setInterval(updateTimer, 1000);

    try {
        // Reset previous state
        await fetch("/api/reset", { method: "POST" });

        // Start generation
        const response = await fetch("/api/generate", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Errore di generazione");
        }

        const { video_id } = await response.json();

        // Start SSE listener
        listenForProgress(video_id);

    } catch (error) {
        showError(error.message);
    }
}

function listenForProgress(videoId) {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("/api/status");

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        progressMessage.textContent = data.message;
        progressPct.textContent = `${data.progress}%`;
        progressFill.style.width = `${data.progress}%`;

        if (data.status === "loading") {
            setStatus("loading", "Caricamento modello...");
        } else if (data.status === "generating") {
            setStatus("loading", `Step ${data.step}/${data.total_steps}`);
        } else if (data.status === "complete") {
            onGenerationComplete(videoId, data.elapsed);
            eventSource.close();
            eventSource = null;
        } else if (data.status === "error") {
            showError(data.message);
            eventSource.close();
            eventSource = null;
        }
    };

    eventSource.onerror = () => {
        // SSE connection closed by server (normal after complete/error)
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    };
}

function onGenerationComplete(videoId, elapsed) {
    clearInterval(generationTimer);
    progressFill.classList.remove("active");
    progressFill.style.width = "100%";
    progressMessage.textContent = `Video generato in ${elapsed}s 🎉`;
    progressPct.textContent = "100%";

    // Show video
    videoPlayer.src = `/api/video/${videoId}`;
    videoOutput.classList.remove("hidden");

    // Setup download
    btnDownload.onclick = () => {
        const a = document.createElement("a");
        a.href = `/api/video/${videoId}`;
        a.download = `videoai_${videoId}.mp4`;
        a.click();
    };

    // Reset button
    btnGenerate.disabled = false;
    btnGenerate.querySelector(".btn-generate-text").textContent = "Genera Video";
    setStatus("idle", "Completato ✅");

    // Auto-scroll to video on mobile
    if (window.innerWidth <= 900) {
        videoOutput.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}

function showError(message) {
    clearInterval(generationTimer);
    progressFill.classList.remove("active");
    progressMessage.textContent = `❌ ${message}`;
    progressPct.textContent = "";
    btnGenerate.disabled = false;
    btnGenerate.querySelector(".btn-generate-text").textContent = "Genera Video";
    setStatus("error", "Errore");
}

function updateTimer() {
    if (!generationStartTime) return;
    const elapsed = Math.floor((Date.now() - generationStartTime) / 1000);
    const min = Math.floor(elapsed / 60);
    const sec = elapsed % 60;
    progressTimer.textContent = `Tempo: ${min}:${sec.toString().padStart(2, "0")}`;
}

btnGenerate.addEventListener("click", startGeneration);

// ─── New Video ───────────────────────────────────────────────────────────────

btnNewVideo.addEventListener("click", () => {
    videoOutput.classList.add("hidden");
    progressSection.classList.add("hidden");
    videoPlayer.src = "";
    removeImage();
    setStatus("idle", "Pronto");
});

// ─── Status Management ──────────────────────────────────────────────────────

function setStatus(type, text) {
    statusDot.className = "status-dot";
    if (type === "loading") statusDot.classList.add("loading");
    if (type === "error") statusDot.classList.add("error");
    statusLabel.textContent = text;
}

// ─── Init ────────────────────────────────────────────────────────────────────

// Check server health on load
(async function init() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.model_loaded) {
            setStatus("idle", "Modello pronto");
        } else {
            setStatus("idle", "Pronto");
        }
    } catch {
        setStatus("error", "Server offline");
    }
})();
