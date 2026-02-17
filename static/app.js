/**
 * app.js — VideoAi Frontend Logic
 * Handles prompt input, image upload, parameter controls, API communication, and video playback.
 * Supports text-to-video (CogVideoX) and image-to-video (SVD).
 */

// ─── DOM Elements ────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const uploadZone = $("#uploadZone");
const uploadContent = $("#uploadContent");
const imagePreview = $("#imagePreview");
const btnRemove = $("#btnRemove");
const fileInput = $("#fileInput");
const btnGenerate = $("#btnGenerate");
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
const promptInput = $("#promptInput");
const motionGroup = $("#motionGroup");

// Slider controls
const stepsEl = $("#steps");
const stepsValue = $("#stepsValue");
const motionEl = $("#motion");
const motionValue = $("#motionValue");
const seedInput = $("#seed");
const seedLabel = $("#seedLabel");

// State
let uploadedFile = null;
let generationTimer = null;
let generationStartTime = null;
let eventSource = null;

// ─── Toggle Groups ──────────────────────────────────────────────────────────

function getToggleValue(groupId) {
    const group = document.getElementById(groupId);
    const active = group.querySelector(".toggle-btn.active");
    return active ? active.dataset.value : null;
}

function setupToggleGroup(groupId) {
    const group = document.getElementById(groupId);
    group.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            updateUIState();
        });
    });
}

setupToggleGroup("resolutionGroup");
setupToggleGroup("durationGroup");
setupToggleGroup("fpsGroup");

// ─── Slider Binding ──────────────────────────────────────────────────────────

stepsEl.addEventListener("input", () => {
    stepsValue.textContent = stepsEl.value;
});

motionEl.addEventListener("input", () => {
    motionValue.textContent = motionEl.value;
});

// New Sliders
const guidanceEl = $("#guidance");
const guidanceValue = $("#guidanceValue");
const strengthEl = $("#strength");
const strengthValue = $("#strengthValue");

guidanceEl.addEventListener("input", () => {
    guidanceValue.textContent = guidanceEl.value;
});

strengthEl.addEventListener("input", () => {
    strengthValue.textContent = strengthEl.value;
});


seedInput.addEventListener("input", () => {
    seedLabel.textContent = seedInput.value ? seedInput.value : "Random";
});

// ─── Prompt Handling ─────────────────────────────────────────────────────────

promptInput.addEventListener("input", () => {
    updateUIState();
});

function updateUIState() {
    const hasPrompt = promptInput.value.trim().length > 0;
    const hasImage = uploadedFile !== null;

    // Enable generate button if we have prompt or image
    btnGenerate.disabled = !hasPrompt && !hasImage;

    // Show/hide SVD-only controls
    const motionGroup = $("#motionGroup");
    const guidanceGroup = $("#guidanceGroup");
    const strengthGroup = $("#strengthGroup");

    if (hasPrompt) {
        // CogVideoX Mode
        motionGroup.classList.add("hidden");
        guidanceGroup.classList.remove("hidden");

        if (hasImage) {
            // Image + Prompt (CogVideoX I2V)
            strengthGroup.classList.remove("hidden");
        } else {
            // Text Only
            strengthGroup.classList.add("hidden");
        }

    } else {
        // Image Only (SVD)
        if (hasImage) {
            motionGroup.classList.remove("hidden");
            guidanceGroup.classList.add("hidden"); // SVD doesn't use guidance in this UI
            strengthGroup.classList.add("hidden");
        }
    }
}

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
        updateUIState();
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
    fileInput.value = "";
    updateUIState();
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
    const prompt = promptInput.value.trim();
    const hasImage = uploadedFile !== null;

    if (!prompt && !hasImage) return;

    // Hide previous video
    videoOutput.classList.add("hidden");

    // Build FormData
    const formData = new FormData();

    if (prompt) {
        formData.append("prompt", prompt);
    }
    if (hasImage) {
        formData.append("image", uploadedFile);
    }

    // Parameters
    formData.append("resolution", getToggleValue("resolutionGroup") || "480p");
    formData.append("duration", getToggleValue("durationGroup") || "6");
    formData.append("fps", getToggleValue("fpsGroup") || "8");
    formData.append("steps", stepsEl.value);
    formData.append("motion", motionEl.value);
    formData.append("guidance", guidanceEl.value);
    formData.append("strength", strengthEl.value);

    if (seedInput.value) {
        formData.append("seed", seedInput.value);
    }

    // UI: Show progress
    progressSection.classList.remove("hidden");
    progressFill.style.width = "0%";
    progressFill.classList.add("active");
    progressMessage.textContent = prompt ? "Avvio generazione text-to-video..." : "Invio immagine...";
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
            onGenerationComplete(videoId, data.elapsed, data);
            eventSource.close();
            eventSource = null;
        } else if (data.status === "error") {
            showError(data.message);
            eventSource.close();
            eventSource = null;
        }
    };

    eventSource.onerror = () => {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    };
}

progressFill.classList.remove("active");
progressFill.style.width = "100%";
progressMessage.textContent = `Video generato in ${elapsed}s 🎉`;
progressPct.textContent = "100%";

// Validate and show seed if available
if (videoId.includes("_")) {
    // I need to change the API to return the seed, 
    // OR I can parse it if I change video_id format? 
    // Better: The SSE "complete" event should carry the seed.
    // Let's assume data.seed is passed in the future step.
}

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
    promptInput.value = "";
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

(async function init() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        if (data.capabilities && !data.capabilities.text_to_video) {
            // No CUDA GPU — show warning
            promptInput.placeholder = "⚠️ Text-to-video richiede GPU NVIDIA. Usa un'immagine per generare.";
        }

        if (data.model_loaded) {
            setStatus("idle", "Modello pronto");
        } else {
            setStatus("idle", "Pronto");
        }
    } catch {
        setStatus("error", "Server offline");
    }
})();
