param(
    [string]$PythonVersion = "3.12",
    [string]$TorchVersion = "2.12.1"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$environment = Join-Path $project ".venv"
$python = Join-Path $environment "Scripts\python.exe"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found; verify the NVIDIA driver before creating the CUDA environment."
}

if (-not (Test-Path -LiteralPath $python)) {
    py "-$PythonVersion" -m venv $environment
}

& $python -m pip install --upgrade pip
& $python -m pip install "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install -e "${project}[dev]"
& $python -c "import json, torch; assert torch.cuda.is_available(), 'Installed torch cannot use CUDA'; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0)}, indent=2))"

Write-Output "CUDA environment is ready. Activate with: $environment\Scripts\Activate.ps1"
