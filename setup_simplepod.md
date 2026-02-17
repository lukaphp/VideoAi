# 🔧 Setup SimplePod — Guida Passo-Passo

Questa guida ti accompagna nel collegare il tuo Mac al server GPU SimplePod per generare video con SVD.

---

## Step 1: Crea una chiave SSH (solo la prima volta)

Apri il terminale sul Mac e incolla:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/simplepod -N ""
cat ~/.ssh/simplepod.pub
```
**Copia l'output** (inizia con `ssh-ed25519 ...`) — ti servirà nel prossimo step.

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBcIDTqJroKDankR1JFVlCPOlqjHfVQKhAhskVoAg+kl gianluca@MACBOOKPRO.station

---

## Step 2: Configura SimplePod

1. Vai su [simplepod.ai](https://simplepod.ai) → **Settings** → **SSH Keys**
2. Incolla la chiave pubblica copiata allo step 1
3. Torna alla pagina server e **noleggia la RTX 3060**:
   - Template: `pytorch/pytorch:latest`
   - Persistent Volume: ✅ **Sì** (20-30GB)
   - Clicca **Deploy**
4. Attendi che lo stato diventi **Running**
5. **Copia le credenziali SSH** dalla dashboard (saranno tipo):
   ```
   ssh root@<IP> -p <PORTA>
   ```

---

## Step 3: Dammi le credenziali

Una volta che il pod è attivo, scrivimi qui:
```
IP: xxx.xxx.xxx.xxx
Porta SSH: xxxxx
```

Io farò tutto dal tuo Mac:
1. Mi connetto al pod via SSH
2. Clono il repo `github.com/lukaphp/VideoAi`
3. Installo le dipendenze
4. Avvio il server Flask
5. Creo un tunnel SSH per la Web UI

Tu dovrai solo aprire **http://localhost:5001** nel browser e usare l'app come prima — ma 30x più veloce!

---

## Step 4 (alternativa): Fai da solo

Se preferisci farlo manualmente:

```bash
# Dal tuo Mac, connettiti al pod
ssh -i ~/.ssh/simplepod root@<IP> -p <PORTA>

# Sul pod, clona e installa
git clone https://github.com/lukaphp/VideoAi.git
cd VideoAi
pip install -r requirements.txt
pip install opencv-python imageio imageio-ffmpeg
python app.py &

# Esci dal pod
exit

# Dal Mac, apri il tunnel
ssh -i ~/.ssh/simplepod -L 5001:localhost:5001 root@<IP> -p <PORTA> -N
```

Apri **http://localhost:5001** nel browser.

---

## 🔒 Sicurezza
- La chiave SSH è **privata e resta sul tuo Mac** (`~/.ssh/simplepod`)
- Il tunnel SSH **cifra tutto il traffico** tra Mac e pod
- Le immagini che carichi passano solo nel tunnel cifrato
- Ricorda di **spegnere il pod** quando hai finito per non pagare ore inutili
