# 📁 Guida Setup AI Video Locale (Mac M1 8GB)

Questa guida configura un ambiente di generazione video Image-to-Video (I2V) completamente offline, ottimizzato per hardware con risorse limitate. Include sia un workflow manuale (Antigravity) che uno script-based con Web UI.

---

## 🛠 1. Requisiti di Sistema & Privacy
* **Hardware:** Apple Silicon M1 (8GB RAM).
* **Software:** Python 3.10+, pip, Antigravity (opzionale, native macOS App).
* **Privacy:** Una volta scaricati i modelli, disattivare il Wi-Fi per garantire che nessun dato (sorgente o target) lasci il dispositivo.

---

## ⚙️ 2. Configurazione Modelli (Context)
Per far girare video con soli 8GB, è fondamentale caricare la versione corretta dei pesi.

### Modello Consigliato: Stable Video Diffusion (SVD) Quantizzato
1.  Apri il **Model Manager** in Antigravity.
2.  Cerca: `SVD-XT` o `SVD-1.1`.
3.  **IMPORTANTE:** Seleziona la versione **4-bit** o **8-bit** (spesso indicata come *Quantized*).
    * *Nota:* Un modello standard occupa ~9GB (Crash garantito). Un modello 4-bit occupa ~3.5GB (Funzionante).

---

## 🧪 3. Workflow Operativo — App Antigravity (Passo dopo Passo)

### Fase A: Preparazione Ambiente (Massimizzare RAM)
1.  **Kill-all:** Chiudi Browser, Spotify, Discord e app in background.
2.  **Activity Monitor:** Apri "Monitoraggio Attività" > Tab Memoria. Controlla che la "Pressione della memoria" sia verde.
3.  **Offline Mode:** Disabilita il Wi-Fi.

### Fase B: Caricamento Immagine
1.  Trascina l'immagine sorgente nel modulo **Image Input**.
2.  Assicurati che l'immagine sia già ridimensionata (es. 512x512) prima di caricarla per risparmiare cicli di calcolo.

### Fase C: Parametri di Generazione (Safety Preset)
Copia e incolla questi valori per evitare il blocco del sistema:

| Parametro | Valore | Note |
| :--- | :--- | :--- |
| **Resolution** | `384 x 640` | Rapporto 9:16 leggero. |
| **Sampling Steps** | `20` | Equilibrio perfetto qualità/tempo. |
| **Motion Bucket Id** | `100` | Movimento fluido ma coerente. |
| **FPS** | `6` | Standard per preview veloci. |
| **Video Frames** | `14` | Massimo gestibile con 8GB. |
| **Scheduler** | `Euler A` | Il più veloce su architettura Metal. |

---

## 🚀 4. Esecuzione (Antigravity)
1.  Clicca su **"Generate"**.
2.  **NON** toccare il computer durante la fase di "Load Model" (i primi 60-90 secondi).
3.  Il video finale verrà salvato automaticamente nella tua cartella locale `~/Pictures/Antigravity/Output`.

---

## 🐍 5. Workflow via Script Python (CLI)

### Installazione
```bash
cd ~/VideoAi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Uso Base
```bash
# Safety preset (consigliato per 8GB RAM)
python generate_video.py foto.jpg

# Con parametri personalizzati
python generate_video.py foto.jpg --steps 25 --frames 20 --motion 80

# Risoluzione minima (per evitare crash)
python generate_video.py foto.jpg --width 448 --height 256 --frames 10 --steps 15

# Con seed per risultati riproducibili
python generate_video.py foto.jpg --seed 42
```

### Parametri Disponibili
| Parametro | Default | Range | Descrizione |
| :--- | :--- | :--- | :--- |
| `--width` | 640 | 256-1024 | Larghezza video (multipli di 64) |
| `--height` | 384 | 256-1024 | Altezza video (multipli di 64) |
| `--steps` | 20 | 5-50 | Passi di sampling |
| `--motion` | 100 | 1-255 | Intensità movimento |
| `--fps` | 6 | 2-30 | Frame al secondo |
| `--frames` | 14 | 4-25 | Numero di frame |
| `--seed` | random | 0-999999999 | Seed per riproducibilità |
| `--output` | auto | path | Percorso file di uscita |

> **Nota:** Il primo run scarica il modello SVD-XT (~4GB). Successivamente funziona offline.

---

## 🌐 6. Web UI (Interfaccia Grafica Locale)

Una web app premium per generare video comodamente dal browser, con drag & drop, slider e preview live.

### Avvio
```bash
cd ~/VideoAi
source venv/bin/activate
python app.py
```
Poi apri: **http://localhost:5001**

### Funzionalità
- ✅ Drag & Drop immagini
- ✅ Slider parametri con preset ottimizzati
- ✅ Barra di progresso in tempo reale (SSE)
- ✅ Player video integrato + download MP4
- ✅ Tema dark premium (glassmorphism)
- ✅ Responsive (funziona anche da iPhone/iPad sulla stessa rete)

### Architettura
```
VideoAi/
├── app.py                 # Server Flask
├── generate_video.py      # Engine SVD (CLI + modulo)
├── requirements.txt       # Dipendenze Python
├── static/
│   ├── index.html         # Interfaccia web
│   ├── style.css          # Tema dark glassmorphism
│   └── app.js             # Logica frontend
└── plan.md                # Questa guida
```

---

## 🧠 7. Tips Avanzati

### Benchmark Indicativi (M1 8GB)
| Risoluzione | Frames | Steps | Tempo Stimato |
| :--- | :--- | :--- | :--- |
| 256×448 | 10 | 15 | ~1.5 min |
| 384×640 | 14 | 20 | ~3-5 min |
| 512×512 | 14 | 25 | ~5-8 min |
| 512×768 | 20 | 30 | ⚠️ Possibile crash |

### Ottimizzazione Memoria
1. **SEMPRE** chiudi tutte le app prima di generare.
2. La variabile `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` è già impostata negli script — previene crash OOM.
3. Se il Mac rallenta, abbassa la risoluzione a `256x448` e frames a `10`.
4. Lo script usa `float16` + `attention_slicing` + `forward_chunking` automaticamente.

### Qualità dell'Immagine Sorgente
- Usa immagini **512x512** o **640x384** (già la risoluzione target).
- Evita sfondi troppo complessi — il modello lavora meglio con soggetti semplici e ben illuminati.
- I volti funzionano bene per piccoli movimenti (blinking, head turn).

### Risoluzione Problemi
* **Il Mac si blocca (Spinning Wheel):** Risoluzione troppo alta. Riavvia e scendi a `256×448`.
* **Errore "Out of Memory":** Troppe app aperte o modello non ottimizzato. Verifica con `Activity Monitor`.
* **L'immagine sembra "sciogliersi":** Abbassa `--motion` a `40-50`.
* **Errore "MPS backend out of memory":** È già gestito dagli script, ma se persiste prova `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` prima di eseguire.
* **Primo avvio lento:** Normale. Il modello (~4GB) viene scaricato e cachato in `~/.cache/huggingface/`.