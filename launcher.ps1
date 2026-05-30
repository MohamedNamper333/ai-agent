param(
    [ValidateSet("web", "cli", "install")]
    [string]$Mode = "web",
    [string]$Model = "",
    [switch]$Ollama
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

# Activate venv
$venvActivate = "$ProjectDir\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

$extraArgs = @()
if ($Model) { $extraArgs += "--model"; $extraArgs += $Model }
if ($Ollama) { $extraArgs += "--ollama" }

switch ($Mode) {
    "install" {
        $installArgs = @()
        if ($Model) { $installArgs += "-ModelName"; $installArgs += $Model }
        if ($Ollama) { $installArgs += "-UseOllama" }
        & ".\install.ps1" $installArgs
    }
    "cli"   { python main.py --cli @extraArgs }
    "web"   { python main.py --web @extraArgs }
}
