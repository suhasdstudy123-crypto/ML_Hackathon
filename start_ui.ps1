# Laptop 2: build and serve the evaluator dashboard.
# Usage (from the repository root):
#   .\start_ui.ps1 -ApiUrl http://192.168.1.20:8000
# -ApiUrl is Laptop 1's address as printed by start_api.ps1. It is pre-filled in the dashboard.
# If PowerShell blocks scripts, run once:  Set-ExecutionPolicy -Scope Process Bypass
param(
    [string]$ApiUrl = "http://localhost:8000",
    [int]$Port = 5173
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Build copies only browser assets (HTML, CSS, JS and the bundled Leaflet) into evaluator_ui\dist.
node evaluator_ui\scripts\build.mjs
if ($LASTEXITCODE -ne 0) { Write-Host "Dashboard build failed." -ForegroundColor Red; exit 1 }

$encoded = [uri]::EscapeDataString($ApiUrl)
Write-Host "Open: http://localhost:$Port/?api=$encoded" -ForegroundColor Green
Write-Host "Laptop 1 must have been started with -UiOrigin http://<this laptop's IP>:$Port" -ForegroundColor Cyan

# Serve only the built folder, never the repository root (which contains protected .route files).
python -m http.server $Port --bind 0.0.0.0 --directory evaluator_ui\dist
