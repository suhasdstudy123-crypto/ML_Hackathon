# Laptop 1: start the PathSnap API (model + HMM + server-side evaluation).
# Usage (from the repository root):
#   .\start_api.ps1 -UiOrigin http://192.168.1.21:5173
# -UiOrigin is the exact address of the dashboard on Laptop 2 (scheme, IP and port).
# If PowerShell blocks scripts, run once:  Set-ExecutionPolicy -Scope Process Bypass
param(
    [string]$UiOrigin = "http://localhost:5173",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Fail early with a clear message if the data or model are missing.
if (-not (Test-Path "data\authoritative\student_test_features.parquet")) {
    Write-Host "data\authoritative is empty. Run: python scripts\import_dataset.py <folder or zip with the seven artifacts>" -ForegroundColor Red
    exit 1
}
python scripts\validate_model.py
if ($LASTEXITCODE -ne 0) { Write-Host "Model validation failed; see the output above." -ForegroundColor Red; exit 1 }

# Some laptops already use 8000 (e.g. the SecuGen fingerprint service). Fail with a clear hint.
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = (Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Write-Host "Port $Port is already used by '$owner'. Re-run with another port, e.g. .\start_api.ps1 -UiOrigin $UiOrigin -Port 8011" -ForegroundColor Red
    exit 1
}

# Always allow the local dashboard too, so the API laptop can also test the UI itself.
$env:PATHSNAP_CORS_ORIGINS = "$UiOrigin,http://localhost:5173,http://127.0.0.1:5173"
$env:PATHSNAP_NOTEBOOK_DEMO_MODE = "1"

$ips = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }).IPAddress
Write-Host "Trusted UI origin(s): $env:PATHSNAP_CORS_ORIGINS" -ForegroundColor Cyan
foreach ($ip in $ips) { Write-Host "Enter this in the dashboard's Backend API URL: http://${ip}:$Port" -ForegroundColor Green }

# Single worker: jobs live in this process's memory.
python -m uvicorn api_server:app --host 0.0.0.0 --port $Port --workers 1
