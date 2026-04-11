# scripts/build_wasm.ps1 — Build echopose_wasm for browser (Windows)
#
# Requires: wasm-pack (cargo install wasm-pack)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
$Crate = Join-Path $PSScriptRoot ".." "ui" "wasm" "echopose_wasm"
$Out = Join-Path $PSScriptRoot ".." "ui" "wasm" "pkg"

# Resolve relative paths
$Crate = (Resolve-Path (Join-Path $PSScriptRoot "..\ui\wasm\echopose_wasm")).Path
$Out = Join-Path (Split-Path $Crate) "pkg"

Write-Host "==> Building echopose_wasm for browser target..."
wasm-pack build $Crate --target web --out-dir $Out --release

Write-Host "==> WASM build complete: $Out"
Get-ChildItem $Out -Filter "*.wasm" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_  ($([math]::Round($_.Length/1KB, 1)) KB)" }
