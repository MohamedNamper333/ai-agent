param(
    [string]$ModelName = "Qwen",
    [switch]$SkipModel,
    [switch]$UseOllama
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "=== AI Agent - Installation ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
try {
    $pyVersion = python --version
    Write-Host "[OK] Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[...] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "[OK] Virtual environment ready" -ForegroundColor Green

# 3. Activate and install deps
$pip = if ($IsWindows -or $env:OS) {
    "$ProjectDir\.venv\Scripts\pip.exe"
} else {
    "$ProjectDir\.venv\bin\pip"
}

Write-Host "[...] Installing Python dependencies..." -ForegroundColor Yellow
& $pip install --upgrade pip -q
& $pip install -r requirements.txt -q
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# 4. Check Ollama (recommended for Windows)
$ollamaFound = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) {
        $ollamaFound = $true
        Write-Host "[OK] Ollama detected!" -ForegroundColor Green
    }
} catch {
    $ollamaFound = $false
}

if ($ollamaFound -or $UseOllama) {
    Write-Host "[...] Using Ollama backend (recommended for Windows)" -ForegroundColor Yellow
    Write-Host "  No need to download GGUF models - Ollama handles them"
    Write-Host ""
    Write-Host "  Make sure you have a model pulled:" -ForegroundColor Cyan
    Write-Host "  ollama pull qwen2.5:7b" -ForegroundColor White
    Write-Host "  ollama pull llama3.2:3b  (faster, smaller)" -ForegroundColor White

    $configContent = @"
# AI Agent Configuration - Ollama
BACKEND = "ollama"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE = "http://127.0.0.1:11434"
N_CTX = 8192
TEMP = 0.7
MAX_TOKENS = 2048
SYSTEM_PROMPT = "You are a helpful, capable AI assistant. You answer questions accurately and thoroughly."
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
DB_PATH = "memory_store.json"
"@
} else {
    # 5. Download GGUF model (fallback)
    Write-Host "[!] Ollama not detected." -ForegroundColor Yellow
    Write-Host "  Recommend: Install Ollama from https://ollama.com for easy setup" -ForegroundColor Cyan
    Write-Host "  Or continue with direct GGUF model download..." -ForegroundColor Gray
    Write-Host ""

    if (-not $SkipModel) {
        Write-Host "[...] Downloading model..." -ForegroundColor Yellow

        $modelUrl = ""
        $modelFile = ""

        switch ($ModelName) {
            "Qwen" {
                $modelUrl = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
                $modelFile = "models\qwen2.5-7b-instruct-q4_k_m.gguf"
            }
            "Llama3" {
                $modelUrl = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
                $modelFile = "models\llama-3.2-3b-q4_k_m.gguf"
            }
            "DeepSeek" {
                $modelUrl = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
                $modelFile = "models\deepseek-r1-7b-q4_k_m.gguf"
            }
            default {
                Write-Host "[ERROR] Unknown model: $ModelName" -ForegroundColor Red
                Write-Host "Available: Qwen, Llama3, DeepSeek" -ForegroundColor Yellow
                exit 1
            }
        }

        if (-not (Test-Path $modelFile)) {
            Write-Host "  Downloading: $ModelName (~4GB)..." -ForegroundColor Cyan
            $null = New-Item -ItemType Directory -Force -Path "models"

            try {
                $ProgressPreference = 'SilentlyContinue'
                Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile -UseBasicParsing
                $ProgressPreference = 'Continue'
            } catch {
                Write-Host "[ERROR] Download failed. Try manually:" -ForegroundColor Red
                Write-Host "  $modelUrl" -ForegroundColor Yellow
                Write-Host "  Save to: $modelFile" -ForegroundColor Yellow
                exit 1
            }
        }
        Write-Host "[OK] Model ready: $modelFile" -ForegroundColor Green
    }

    $configContent = @"
# AI Agent Configuration - Direct GGUF
BACKEND = "llama"
MODEL_PATH = "$(Get-ChildItem models\*.gguf | Select-Object -First 1 -ExpandProperty Name)"
N_GPU_LAYERS = -1
N_CTX = 8192
N_THREADS = 6
TEMP = 0.7
MAX_TOKENS = 2048
SYSTEM_PROMPT = "You are a helpful, capable AI assistant. You answer questions accurately and thoroughly."
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
DB_PATH = "memory_store.json"
"@
}

Set-Content -Path "config.txt" -Value $configContent

Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run the agent:" -ForegroundColor Green
Write-Host "  .venv\Scripts\activate" -ForegroundColor White
Write-Host "  python main.py --cli     (chat mode)" -ForegroundColor White
Write-Host "  python main.py --web     (web interface)" -ForegroundColor White
