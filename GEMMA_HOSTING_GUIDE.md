# Hosting Gemma 2 9B (Auditor) — Windows & Linux Setup Guide

Gemma 2 9B serves as the auditor (Phase 6) in the Nuclei Template Generation System. It loads after CodeT5 generates a template, reads the template against the original claims, and decides whether the template is valid or should be vetoed.

Gemma never sees a report cold. It always receives: the proposer's claims, the generated YAML template, and the model scores — then looks for reasons the output is wrong.

---

## What Gemma Does in the Pipeline

```
Phase 5: CodeT5 generates YAML template
                    ↓
Phase 6: Gemma loads → reads claims + template → audits → unloads
                    ↓
         Pass → Phase 7 (assembly)
         Veto → template rejected, pipeline stops
```

Gemma checks for:
- **Claim-template mismatch:** proposer said SSRF but template injects SQL syntax
- **Fabricated matchers:** template references paths or parameters not in the original report
- **Evidence gaps:** claims were borderline and template is too aggressive for the confidence level
- **Wrong vuln group:** payload-style template generated for a logic-flaw vulnerability
- **Safety violations:** template exceeds allowed actions for the vuln type

Hard veto rules: if Gemma raises >2 specific objections for RCE or Auth Bypass, the template is rejected regardless of all other scores.

---

## Hardware Requirements

| Resource | Gemma 2 9B Q4_K_M |
|----------|-------------------|
| GPU VRAM | ~5.5GB model + ~0.5-1GB KV cache = ~6-6.5GB |
| System RAM | 8GB minimum (16GB recommended) |
| Storage | ~5.5GB for model weights |
| CPU (no GPU) | 16GB RAM minimum, ~5-10x slower inference |

Gemma is slightly larger than Llama 8B. At Q4_K_M with 4096 context, it fits comfortably in 8GB VRAM. Never loaded simultaneously with Llama — sequential only.

---

## Option A: Ollama (Recommended)

### Windows

#### Step 1: Install Ollama

If you already installed Ollama for Llama, skip to Step 2.

```
1. Download from https://ollama.com/download/windows
2. Run the .exe installer
3. Installation completes silently
```

Verify:

```cmd
ollama --version
```

#### Step 2: Pull Gemma 2 9B

```cmd
ollama pull gemma2:9b
```

Downloads the Q4_K_M quantized version (~5.5GB). Stored in `C:\Users\<you>\.ollama\models\`.

To see what's installed:

```cmd
ollama list
```

Should show both `llama3.1:8b` and `gemma2:9b` if you followed the Llama guide first.

#### Step 3: Test Gemma interactively

```cmd
ollama run gemma2:9b
```

Type a message, get a response. Type `/bye` to exit.

Test with an auditor-style prompt:

```cmd
ollama run gemma2:9b "I claimed this endpoint is vulnerable to SSRF because the 'url' parameter accepts user input and the server fetches external resources. The generated Nuclei template uses a SQL injection payload in the 'id' parameter. Is this template correct?"
```

Gemma should flag the mismatch between the SSRF claim and the SQLi payload.

#### Step 4: Test the API

Ollama runs an API server automatically on port 11434.

```cmd
curl http://localhost:11434/api/generate -d "{\"model\": \"gemma2:9b\", \"prompt\": \"Review this security claim and identify problems: vuln_type=XSS, confidence=0.82, but the template matcher checks for SQL error strings.\", \"stream\": false}"
```

#### Step 5: Test with structured auditor prompt

This mirrors what the pipeline actually sends to Gemma:

```cmd
curl http://localhost:11434/api/generate -d "{\"model\": \"gemma2:9b\", \"prompt\": \"You are a skeptical security reviewer. Your job is to find flaws in this assessment.\n\nPROPOSER CLAIMS:\n- vuln_type: path_traversal\n- confidence: 0.68\n- parameter: filename\n- evidence: parameter accepts file paths, server runs PHP\n\nGENERATED TEMPLATE MATCHERS:\n- word: 'root:x:0:0'\n- status: 200\n- part: body\n\nDoes this template correctly test the claimed vulnerability? List any objections.\", \"stream\": false}"
```

#### Step 6: Python integration

```cmd
pip install ollama
```

```python
import ollama
import json

def audit_template(claims: dict, template_yaml: str) -> dict:
    """
    Phase 6: Gemma audits a generated template against proposer claims.
    Returns audit verdict with agreement status and objections.
    """
    prompt = f"""You are a skeptical security reviewer. Your default position is disagreement.
Only agree if evidence strongly supports the template.

PROPOSER CLAIMS:
{json.dumps(claims, indent=2)}

GENERATED NUCLEI TEMPLATE:
{template_yaml}

Evaluate this template against the claims. Respond ONLY with JSON:
{{
    "agreement": true/false,
    "objections": ["list of specific problems found"],
    "missing_evidence": ["evidence that should exist but doesn't"],
    "recommended_constraints": ["safety limits to add"],
    "confidence_appropriate": true/false
}}"""

    response = ollama.generate(
        model='gemma2:9b',
        prompt=prompt,
        options={
            'temperature': 0.1,
            'num_predict': 1024,
            'num_ctx': 4096,
        }
    )
    return response['response']


# Example usage
claims = {
    "vuln_type": "ssrf",
    "confidence": 0.71,
    "parameter": "url",
    "evidence": {
        "param_accepts_urls": True,
        "server_side_fetch": "probable"
    },
    "allowed_tests": ["oob_dns"]
}

template = """
id: ssrf-url-param
info:
  name: SSRF via url parameter
  severity: high
http:
  - method: GET
    path:
      - "{{BaseURL}}/fetch?url=https://{{interactsh-url}}"
    matchers:
      - type: word
        part: interactsh_protocol
        words:
          - "dns"
"""

verdict = audit_template(claims, template)
print(verdict)
```

---

### Linux (Ubuntu/Debian)

#### Step 1: Install Ollama

If already installed for Llama, skip to Step 2.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
sudo systemctl status ollama
```

#### Step 2: Pull Gemma 2 9B

```bash
ollama pull gemma2:9b
```

Models stored in `~/.ollama/models/`.

#### Step 3: Test

```bash
ollama run gemma2:9b "Review this claim: an endpoint accepts a 'file' parameter and the template tests for /etc/passwd in the response. The claimed vuln type is XSS. Is this correct?"
```

#### Step 4: Verify GPU usage

```bash
# Start Gemma
ollama run gemma2:9b "test"

# In another terminal, check VRAM usage
nvidia-smi
```

You should see `ollama` using ~5.5-6.5GB VRAM. If it's using 0 GPU memory, CUDA isn't configured:

```bash
# Check CUDA
nvcc --version

# Install if missing
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
sudo reboot
```

#### Step 5: API test

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma2:9b",
  "prompt": "You are a skeptical security auditor. The proposer claims IDOR with confidence 0.55. The template increments user IDs and checks for 200 status. What are the problems with this template?",
  "stream": false
}'
```

---

## Option B: llama.cpp (Direct Control)

Use this if you need precise control over GPU layers, context length, or are running Gemma on a separate port from Llama.

### Windows

#### Step 1: Download pre-built binaries

If you already have llama.cpp from the Llama setup, skip to Step 2.

```
1. Go to https://github.com/ggml-org/llama.cpp/releases
2. Download for your GPU:
   - NVIDIA: llama-<version>-bin-win-cuda-cu12.x-x64.zip
   - AMD: llama-<version>-bin-win-vulkan-x64.zip
   - CPU only: llama-<version>-bin-win-avx2-x64.zip
3. Extract to C:\llama-cpp\
```

#### Step 2: Download Gemma 2 9B GGUF

```cmd
pip install huggingface-hub

huggingface-cli download bartowski/gemma-2-9b-it-GGUF gemma-2-9b-it-Q4_K_M.gguf --local-dir C:\llama-cpp\models\
```

Or download manually from https://huggingface.co/bartowski/gemma-2-9b-it-GGUF — grab the Q4_K_M file (~5.5GB).

#### Step 3: Test from command line

```cmd
cd C:\llama-cpp

llama-cli.exe -m models\gemma-2-9b-it-Q4_K_M.gguf -ngl 99 -cnv --chat-template gemma
```

#### Step 4: Run as API server

```cmd
llama-server.exe -m models\gemma-2-9b-it-Q4_K_M.gguf -ngl 99 --host 127.0.0.1 --port 8081 -c 4096
```

Note: port 8081, not 8080. If you're running Llama on 8080, Gemma gets 8081. The pipeline stops the Llama server before starting Gemma's, but separate ports prevent accidents.

Test:

```cmd
curl http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"gemma\", \"messages\": [{\"role\": \"user\", \"content\": \"Audit this template: claims say XSS but matchers check for SQL errors\"}]}"
```

### Linux

#### Step 1: Build llama.cpp (if not already done)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

#### Step 2: Download Gemma 2 9B GGUF

```bash
pip install huggingface-hub

huggingface-cli download bartowski/gemma-2-9b-it-GGUF \
  gemma-2-9b-it-Q4_K_M.gguf \
  --local-dir ~/models/
```

#### Step 3: Test

```bash
./build/bin/llama-cli -m ~/models/gemma-2-9b-it-Q4_K_M.gguf \
  -ngl 99 -cnv --chat-template gemma
```

#### Step 4: Run as API server

```bash
./build/bin/llama-server \
  -m ~/models/gemma-2-9b-it-Q4_K_M.gguf \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8081 \
  -c 4096
```

---

## Pipeline Integration: Auditor Wrapper

This is the production wrapper that goes in `llm/auditor.py`. It handles loading Gemma, sending the structured audit prompt, parsing the verdict, and applying veto logic.

### With Ollama

```python
import ollama
import json
from core.types import Claims, AuditVerdict, TemplateArtifact

class GemmaAuditor:
    """Phase 6: Gemma 2 9B audits generated templates against proposer claims."""

    MODEL = 'gemma2:9b'

    # Vuln types where hard veto applies (>2 objections = reject)
    HARD_VETO_TYPES = {'rce', 'auth_bypass'}

    def __init__(self):
        self.options = {
            'temperature': 0.1,
            'num_predict': 1024,
            'num_ctx': 4096,
        }

    def _build_prompt(self, claims: Claims, template_yaml: str) -> str:
        return f"""You are a skeptical security reviewer auditing a generated Nuclei template.
Your default position is: this template is wrong until proven otherwise.

ORIGINAL PROPOSER CLAIMS:
- Vulnerability type: {claims.vuln_type}
- Confidence: {claims.confidence}
- Target parameter: {claims.parameter}
- Evidence: {json.dumps(claims.evidence)}
- Allowed tests: {claims.allowed_tests}

GENERATED NUCLEI TEMPLATE:
{template_yaml}

CHECK EACH OF THE FOLLOWING:
1. Does the template test the correct vulnerability type?
2. Do the matchers correspond to evidence in the claims?
3. Are request paths and parameters real (not fabricated)?
4. Is the template's aggressiveness appropriate for the confidence level?
5. Does the template stay within the allowed test boundaries?
6. Are there alternative explanations for the signals described?

Respond ONLY with JSON matching this exact schema:
{{
    "agreement": false,
    "objection_count": 0,
    "objections": [
        {{"type": "mismatch|fabricated|missing_evidence|too_aggressive|wrong_vuln_type|safety", "detail": "specific description"}}
    ],
    "missing_evidence": ["what evidence should exist but doesn't"],
    "recommended_constraints": ["safety limits to apply"],
    "confidence_appropriate": false
}}"""

    def audit(self, claims: Claims, template_yaml: str) -> AuditVerdict:
        """Audit a single template. Returns verdict with objections."""
        prompt = self._build_prompt(claims, template_yaml)

        response = ollama.generate(
            model=self.MODEL,
            prompt=prompt,
            options=self.options
        )

        # Parse Gemma's response
        try:
            verdict_raw = json.loads(response['response'])
        except json.JSONDecodeError:
            # If Gemma doesn't return valid JSON, treat as objection
            return AuditVerdict(
                agreement=False,
                objections=[{"type": "parse_error", "detail": "Auditor response was not valid JSON"}],
                hard_veto=True
            )

        # Apply hard veto logic
        objection_count = len(verdict_raw.get('objections', []))
        hard_veto = False

        if claims.vuln_type in self.HARD_VETO_TYPES and objection_count > 2:
            hard_veto = True
        elif objection_count > 3:
            hard_veto = True

        return AuditVerdict(
            agreement=verdict_raw.get('agreement', False),
            objections=verdict_raw.get('objections', []),
            missing_evidence=verdict_raw.get('missing_evidence', []),
            recommended_constraints=verdict_raw.get('recommended_constraints', []),
            confidence_appropriate=verdict_raw.get('confidence_appropriate', False),
            hard_veto=hard_veto
        )

    def batch_audit(self, items: list[tuple[Claims, str]]) -> list[AuditVerdict]:
        """
        Batch audit multiple templates.
        Gemma stays loaded for the entire batch — one load, N audits, one unload.
        """
        return [self.audit(claims, template) for claims, template in items]

    def unload(self):
        """Force-unload Gemma from VRAM to free memory for other models."""
        import requests
        requests.post('http://localhost:11434/api/generate', json={
            'model': self.MODEL,
            'keep_alive': 0
        })
```

### With llama.cpp

```python
import subprocess
import time
import requests
import json
from core.types import Claims, AuditVerdict

class GemmaAuditorLlamaCpp:
    """Phase 6: Gemma 2 9B via llama-server process."""

    HARD_VETO_TYPES = {'rce', 'auth_bypass'}

    def __init__(self, model_path: str, server_bin: str, port: int = 8081):
        self.model_path = model_path
        self.server_bin = server_bin
        self.port = port
        self.process = None
        self.base_url = f"http://127.0.0.1:{port}"

    def load(self):
        """Start llama-server with Gemma."""
        self.process = subprocess.Popen([
            self.server_bin,
            '-m', self.model_path,
            '-ngl', '99',
            '--host', '127.0.0.1',
            '--port', str(self.port),
            '-c', '4096',
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for server ready
        for _ in range(30):
            try:
                requests.get(f"{self.base_url}/health")
                return
            except requests.ConnectionError:
                time.sleep(1)
        raise RuntimeError("Gemma server failed to start within 30 seconds")

    def unload(self):
        """Stop the server, free VRAM."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            time.sleep(2)  # let GPU memory release

    def audit(self, claims: Claims, template_yaml: str) -> AuditVerdict:
        """Send audit request to running Gemma server."""
        if not self.process:
            self.load()

        prompt = self._build_prompt(claims, template_yaml)

        response = requests.post(f"{self.base_url}/v1/chat/completions", json={
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 1024,
        })

        content = response.json()['choices'][0]['message']['content']

        try:
            verdict_raw = json.loads(content)
        except json.JSONDecodeError:
            return AuditVerdict(agreement=False, objections=[{"type": "parse_error"}], hard_veto=True)

        objection_count = len(verdict_raw.get('objections', []))
        hard_veto = (claims.vuln_type in self.HARD_VETO_TYPES and objection_count > 2) or objection_count > 3

        return AuditVerdict(
            agreement=verdict_raw.get('agreement', False),
            objections=verdict_raw.get('objections', []),
            hard_veto=hard_veto
        )

    def batch_audit(self, items: list[tuple[Claims, str]]) -> list[AuditVerdict]:
        """Load once, audit all, unload once."""
        self.load()
        results = [self.audit(claims, template) for claims, template in items]
        self.unload()
        return results

    def _build_prompt(self, claims, template_yaml):
        # Same prompt builder as Ollama version
        ...
```

---

## Auditor Prompt Templates

Each vuln type gets its own auditor prompt in `llm/prompts/auditor/`. The base structure is the same but the specific checks differ.

### Example: `llm/prompts/auditor/sqli.txt`

```
You are a skeptical security reviewer auditing a Nuclei template for SQL Injection.

CHECK SPECIFICALLY:
- Does the template use boolean-based or time-based blind techniques only? (UNION extraction is forbidden)
- Are payloads from the approved payload family? (no custom SQL beyond the allowlist)
- Do matchers look for SQL error signatures or time differentials, NOT extracted data?
- Is the target parameter actually user-controllable based on the claims?
- Does the tech stack support the payload type? (don't send MySQL payloads to a MongoDB backend)
```

### Example: `llm/prompts/auditor/idor.txt`

```
You are a skeptical security reviewer auditing a Nuclei template for IDOR.

CHECK SPECIFICALLY:
- Does the template test ID increment/decrement only? (no bulk enumeration)
- Are there two requests: baseline (own resource) and mutated (other user's resource)?
- Does the matcher compare responses between baseline and mutated, not just check for 200?
- Is the object reference pattern realistic based on the claims? (sequential integer, UUID, etc.)
- Does the template avoid any actions that would modify or delete resources?
```

### Example: `llm/prompts/auditor/rce.txt`

```
You are a skeptical security reviewer auditing a Nuclei template for Remote Code Execution.
THIS IS A HIGH-SEVERITY VULN TYPE. Apply maximum scrutiny. Hard veto threshold: >2 objections.

CHECK SPECIFICALLY:
- Does the template use ONLY OOB callbacks or benign echo commands (id, whoami, echo nuclei-test)?
- Are there ANY arbitrary commands, file writes, reverse shells, or file reads? (REJECT IMMEDIATELY)
- Is the detection method OOB DNS/HTTP preferred over inline response matching?
- Does the tech stack and entry point actually support code execution based on the claims?
- Is the confidence level high enough to justify testing for RCE? (reject if proposer confidence < 0.6)
```

---

## Gemma Configuration for Auditing

### Temperature: 0.1

Low temperature is critical. Gemma is making a judgment call, not generating creative text. Higher temperature introduces randomness into audit verdicts — you want the same template audited the same way every time.

### Context length: 4096

The audit prompt includes: system instructions (~300 tokens), claims (~200 tokens), full YAML template (~500-1500 tokens), and output format (~200 tokens). 4096 handles this comfortably. If templates get very long, increase to 8192 but expect higher VRAM usage.

### num_predict / max_tokens: 1024

Audit responses are structured JSON — objections, evidence gaps, constraints. 1024 tokens is plenty. Don't set this higher or Gemma may ramble.

---

## Monitoring Gemma's Performance

Track these metrics to know if Gemma is doing its job:

| Metric | Target | Action if Off |
|--------|--------|---------------|
| Agreement rate | 60-75% | >90% = auditor too lenient (loosen prompt bias), <40% = proposer too noisy |
| Hard veto rate | 5-15% | >30% = pipeline has a quality problem upstream, <2% = auditor not catching issues |
| Parse error rate | <5% | >10% = prompt format instructions need tightening |
| Avg objections per audit | 1-3 | >5 = Gemma is nitpicking, 0 = not scrutinizing |
| RCE/Auth Bypass veto rate | 15-30% | Should be higher than other types — these are gated more strictly |

### Logging audits for review

Every audit verdict should be logged with:

```python
audit_log = {
    'artifact_id': artifact_id,
    'vuln_type': claims.vuln_type,
    'proposer_confidence': claims.confidence,
    'gemma_agreement': verdict.agreement,
    'objection_count': len(verdict.objections),
    'objection_types': [o['type'] for o in verdict.objections],
    'hard_veto': verdict.hard_veto,
    'timestamp': datetime.utcnow().isoformat()
}
```

After 200+ audits, review the logs. If Gemma consistently objects to the same objection type (e.g., "fabricated" for IDOR templates), the problem is CodeT5's logic generator, not Gemma.

---

## Troubleshooting

### Gemma returns garbage / not JSON

Tighten the prompt. Add at the end:

```
CRITICAL: Your entire response must be valid JSON. No preamble, no markdown, no explanation outside the JSON object. Start with { and end with }.
```

If still failing, try wrapping the expected output in a code fence in the prompt so Gemma recognizes the format.

### Gemma agrees with everything

The auditor prompt is too soft. Add stronger adversarial framing:

```
Your job is to REJECT templates. Assume every template is wrong. Only agree if you cannot find a single flaw after checking all criteria. When in doubt, disagree.
```

### Gemma rejects everything

The auditor prompt is too aggressive, or the proposer confidence threshold is too low (sending weak claims through). Check:
- Are proposer confidence values reasonable? (>0.6 for most types)
- Is the fusion engine's emit threshold correct? (0.65)
- Is Gemma's prompt demanding evidence that doesn't exist in the claims schema?

### VRAM error when loading Gemma

Llama wasn't fully unloaded. Force unload:

```bash
# Ollama
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b", "keep_alive": 0}'

# Wait 5 seconds, then load Gemma
ollama run gemma2:9b "test"
```

### Slow audit times

Expected: 5-15 seconds per audit on GPU, 30-60 seconds on CPU. If slower:
- Check GPU is being used (`nvidia-smi`)
- Reduce context length if templates are short (`-c 2048`)
- Batch all audits together to avoid repeated model loads

---

## Quick Start Checklist

```bash
# Already have Ollama installed from Llama setup

# 1. Pull Gemma
ollama pull gemma2:9b

# 2. Test interactively
ollama run gemma2:9b "You are a security auditor. Is this XSS template correct if it checks for SQL errors?"

# 3. Test API
curl http://localhost:11434/api/generate -d '{
  "model": "gemma2:9b",
  "prompt": "Audit this: claims say SSRF, template sends SQLi payload",
  "stream": false
}'

# 4. Install Python client
pip install ollama

# Done. Auditor ready for Phase 6.
```
