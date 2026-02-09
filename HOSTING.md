# Hosting Gemma 2 9B — Windows & Linux Setup Guide

How to download, install, and run Gemma 2 9B locally on your machine.

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 6GB | 8GB |
| System RAM | 8GB (GPU) / 16GB (CPU-only) | 16-32GB |
| Storage | 6GB free | 10GB free |

Gemma 2 9B at Q4_K_M quantization is ~5.5GB. No GPU required — it runs on CPU using system RAM, just 5-10x slower.

---

## Option A: Ollama (Simplest)

### Windows

**Install:**

```
1. Download from https://ollama.com/download/windows
2. Run the .exe installer
3. Installation completes silently
```

**Verify and pull model:**

```cmd
ollama --version
ollama pull gemma2:9b
```

**Run:**

```cmd
ollama run gemma2:9b
```

Type messages, get responses. `/bye` to exit.

**API access:**

Ollama runs an API server automatically on port 11434.

```cmd
curl http://localhost:11434/api/generate -d "{\"model\": \"gemma2:9b\", \"prompt\": \"Explain what CORS is.\", \"stream\": false}"
```

**Python:**

```cmd
pip install ollama
```

```python
import ollama

response = ollama.generate(model='gemma2:9b', prompt='Explain what CORS is.')
print(response['response'])
```

---

### Linux (Ubuntu/Debian)

**Install:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify and pull model:**

```bash
ollama --version
sudo systemctl status ollama    # should show active
ollama pull gemma2:9b
```

**Run:**

```bash
ollama run gemma2:9b
```

**API access:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma2:9b",
  "prompt": "Explain what CORS is.",
  "stream": false
}'
```

**Verify GPU is being used:**

```bash
ollama run gemma2:9b "test"
# In another terminal:
nvidia-smi    # should show ollama process using ~5.5GB VRAM
```

If GPU isn't detected:

```bash
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
sudo reboot
```

**Python:**

```bash
pip install ollama
```

```python
import ollama

response = ollama.generate(model='gemma2:9b', prompt='Explain what CORS is.')
print(response['response'])
```

---

## Option B: llama.cpp (More Control)

### Windows

**Step 1: Get llama.cpp**

```
1. Go to https://github.com/ggml-org/llama.cpp/releases
2. Download for your hardware:
   - NVIDIA GPU: llama-<version>-bin-win-cuda-cu12.x-x64.zip
   - AMD GPU: llama-<version>-bin-win-vulkan-x64.zip
   - CPU only: llama-<version>-bin-win-avx2-x64.zip
3. Extract to C:\llama-cpp\
```

**Step 2: Download the model**

```cmd
pip install huggingface-hub

huggingface-cli download bartowski/gemma-2-9b-it-GGUF gemma-2-9b-it-Q4_K_M.gguf --local-dir C:\llama-cpp\models\
```

Or download manually from https://huggingface.co/bartowski/gemma-2-9b-it-GGUF (grab the Q4_K_M file, ~5.5GB).

**Step 3: Run interactively**

```cmd
cd C:\llama-cpp
llama-cli.exe -m models\gemma-2-9b-it-Q4_K_M.gguf -ngl 99 -cnv --chat-template gemma
```

`-ngl 99` puts all layers on GPU. Drop to `-ngl 0` for CPU-only.

**Step 4: Run as API server**

```cmd
llama-server.exe -m models\gemma-2-9b-it-Q4_K_M.gguf -ngl 99 --host 0.0.0.0 --port 8080 -c 4096
```

Test:

```cmd
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"gemma\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain what CORS is.\"}]}"
```

**Python (OpenAI-compatible client):**

```cmd
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Explain what CORS is."}]
)
print(response.choices[0].message.content)
```

---

### Linux (Ubuntu/Debian)

**Step 1: Build llama.cpp**

```bash
sudo apt update && sudo apt install -y build-essential cmake git

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON    # NVIDIA GPU
# cmake -B build                  # CPU only
# cmake -B build -DGGML_VULKAN=ON # AMD GPU
cmake --build build --config Release -j$(nproc)
```

**Step 2: Download the model**

```bash
pip install huggingface-hub

huggingface-cli download bartowski/gemma-2-9b-it-GGUF \
  gemma-2-9b-it-Q4_K_M.gguf \
  --local-dir ~/models/
```

**Step 3: Run interactively**

```bash
./build/bin/llama-cli -m ~/models/gemma-2-9b-it-Q4_K_M.gguf \
  -ngl 99 -cnv --chat-template gemma
```

**Step 4: Run as API server**

```bash
./build/bin/llama-server \
  -m ~/models/gemma-2-9b-it-Q4_K_M.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096
```

---

## Quantization Options

| Quantization | Size | Quality Loss | Use When |
|-------------|------|-------------|----------|
| Q8_0 | ~9.5GB | Minimal | 12GB+ VRAM |
| Q6_K | ~7.4GB | Very small | 10GB VRAM |
| Q5_K_M | ~6.4GB | Small | 8GB VRAM (tight) |
| **Q4_K_M** | **~5.5GB** | **Acceptable** | **8GB VRAM (recommended)** |
| Q3_K_M | ~4.3GB | Significant | 6GB VRAM |
| Q2_K | ~3.5GB | Major | 4GB VRAM (last resort) |

---

## Key Options

| Option | Ollama | llama.cpp | What It Does |
|--------|--------|-----------|-------------|
| GPU layers | automatic | `-ngl 99` | How many layers run on GPU. 99 = all. |
| Context length | `--num-ctx 4096` | `-c 4096` | Max input+output tokens. Higher = more VRAM. |
| Temperature | via options dict | via API param | Lower = more deterministic. |
| CPU only | automatic fallback | `-ngl 0` | Runs on RAM instead of VRAM. |

---

## Troubleshooting

**CUDA out of memory:** Lower GPU layers (`-ngl 30` instead of 99) or use a smaller quantization (Q3_K_M).

**Slow generation:** Verify GPU is being used with `nvidia-smi`. If VRAM shows 0, CUDA drivers need installing.

**Ollama won't start (Windows):**

```cmd
tasklist | findstr ollama
taskkill /f /im ollama.exe
ollama serve
```

**Model not found:**

```bash
ollama list              # check what's installed
ollama pull gemma2:9b    # re-pull if missing
```

---

## Quick Start

```bash
# Ollama (fastest path)
# Windows: download installer from https://ollama.com/download/windows
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull gemma2:9b
ollama run gemma2:9b "Hello, what can you do?"

# Done. ~10 minutes from zero to running.
```
