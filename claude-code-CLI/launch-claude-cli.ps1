# launch-claude-cli.ps1
# Run this from your claude-code-CLI directory

$CLI_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Environment variables ──────────────────────────────────────────────────────
$env:ANTHROPIC_API_KEY       = "ccnim"
$env:ANTHROPIC_BASE_URL      = "http://127.0.0.1:8082"
$env:CLAUDE_DISABLE_AUTOUPDATE = "1"
$env:DISABLE_AUTOUPDATER     = "1"
# DEBUG=* is intentionally omitted — it floods stdout with noise from every subsystem.
# Set $env:DEBUG = "claude:*" for targeted Claude debugging if needed.
# Prevent ripgrep spawn errors by pointing to a known location
$env:RIPGREP_PATH            = "$CLI_DIR\node_modules\@vscode\ripgrep\bin\rg.exe"

# ── Ripgrep: download if missing ──────────────────────────────────────────────
$rgVendorPath = "$CLI_DIR\src\utils\vendor\ripgrep\x64-win32\rg.exe"

if (-not (Test-Path $rgVendorPath)) {
    Write-Host "ripgrep not found at vendor path — attempting to fix..." -ForegroundColor Yellow

    # Try node_modules location first
    $rgNodePath = "$CLI_DIR\node_modules\@vscode\ripgrep\bin\rg.exe"
    if (Test-Path $rgNodePath) {
        $rgDir = Split-Path -Parent $rgVendorPath
        New-Item -ItemType Directory -Force -Path $rgDir | Out-Null
        Copy-Item $rgNodePath $rgVendorPath
        Write-Host "Copied rg.exe from node_modules → vendor path." -ForegroundColor Green
    } else {
        # Download ripgrep binary directly
        Write-Host "Downloading ripgrep..." -ForegroundColor Cyan
        $rgVersion  = "14.1.0"
        $rgUrl      = "https://github.com/BurntSushi/ripgrep/releases/download/$rgVersion/ripgrep-$rgVersion-x86_64-pc-windows-msvc.zip"
        $rgZip      = "$env:TEMP\ripgrep.zip"
        $rgExtract  = "$env:TEMP\ripgrep_extract"

        Invoke-WebRequest -Uri $rgUrl -OutFile $rgZip -UseBasicParsing
        Expand-Archive -Path $rgZip -DestinationPath $rgExtract -Force

        $rgBin = Get-ChildItem -Path $rgExtract -Recurse -Filter "rg.exe" | Select-Object -First 1
        if ($rgBin) {
            $rgDir = Split-Path -Parent $rgVendorPath
            New-Item -ItemType Directory -Force -Path $rgDir | Out-Null
            Copy-Item $rgBin.FullName $rgVendorPath
            Write-Host "ripgrep downloaded and placed at: $rgVendorPath" -ForegroundColor Green
        } else {
            Write-Warning "Could not find rg.exe in downloaded archive. CLI may still run without ripgrep."
        }
        Remove-Item $rgZip -ErrorAction SilentlyContinue
        Remove-Item $rgExtract -Recurse -ErrorAction SilentlyContinue
    }
}

# ── Patch autoUpdater.ts if it has bad exports ────────────────────────────────
$autoUpdaterPath = "$CLI_DIR\src\utils\autoUpdater.ts"
$stubMissing = $false

if (Test-Path $autoUpdaterPath) {
    $content = Get-Content $autoUpdaterPath -Raw
    if ($content -notmatch "checkGlobalInstallPermissions" -or $content -notmatch "getMaxVersion") {
        $stubMissing = $true
    }
} else {
    $stubMissing = $true
}

if ($stubMissing) {
    Write-Host "Patching autoUpdater.ts with complete stub..." -ForegroundColor Yellow
    $stub = @'
// STUB: autoUpdater.ts
export async function checkForUpdates(): Promise<void> { return; }
export async function checkGlobalInstallPermissions(): Promise<boolean> { return true; }
export function getMaxVersion(): string { return "999.999.999"; }
export async function installUpdate(): Promise<void> { return; }
export async function downloadUpdate(): Promise<void> { return; }
export function isUpdateAvailable(): boolean { return false; }
export async function getLatestVersion(): Promise<string> { return "999.999.999"; }
export function getCurrentVersion(): string { return "0.0.0"; }
export async function performUpdate(): Promise<void> { return; }
export function shouldCheckForUpdate(): boolean { return false; }
export async function notifyUserOfUpdate(): Promise<void> { return; }
export const autoUpdater = {
  checkForUpdates, installUpdate, downloadUpdate, isUpdateAvailable,
  getLatestVersion, getCurrentVersion, performUpdate, shouldCheckForUpdate,
  notifyUserOfUpdate, checkGlobalInstallPermissions, getMaxVersion,
};
export default autoUpdater;
'@
    Set-Content -Path $autoUpdaterPath -Value $stub -Encoding UTF8
    Write-Host "autoUpdater.ts patched." -ForegroundColor Green
}

# ── Launch CLI ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Starting Claude Code CLI..." -ForegroundColor Cyan
Write-Host "  Base URL : $env:ANTHROPIC_BASE_URL"
Write-Host "  API Key  : $env:ANTHROPIC_API_KEY"
Write-Host ""

Set-Location $CLI_DIR

# Try bun source first, fall back to dist
if (Test-Path "$CLI_DIR\src\entrypoints\cli.tsx") {
    Write-Host "Running: bun run src/entrypoints/cli.tsx" -ForegroundColor DarkGray
    & bun run src/entrypoints/cli.tsx @args
} elseif (Test-Path "$CLI_DIR\dist\cli.mjs") {
    Write-Host "Running: node dist/cli.mjs" -ForegroundColor DarkGray
    & node dist/cli.mjs @args
} else {
    Write-Error "Could not find CLI entry point. Make sure you're in the claude-code-CLI directory."
    exit 1
}