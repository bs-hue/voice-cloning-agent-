param(
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$distIndex = Join-Path $root "apps\web\dist\index.html"
$runScript = Join-Path $root "apps\api\run_server.sh"

if (-not (Test-Path $distIndex)) {
    Write-Host "Building SwaraLab web frontend..."
    $env:NPM_CONFIG_CACHE = "E:\npm-cache"
    $env:TEMP = "E:\tmp"
    $env:TMP = "E:\tmp"
    Push-Location (Join-Path $root "apps\web")
    npm run build
    Pop-Location
}

(Get-Content $runScript -Raw) -replace "`r`n", "`n" | Set-Content $runScript -NoNewline

if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://127.0.0.1:8000"
    } | Out-Null
}

Write-Host "Starting SwaraLab local dashboard at http://127.0.0.1:8000 ..."
wsl -d SwaraLabWSL -e bash "/mnt/e/Antigravity project/voice cloning agent/apps/api/run_server.sh"

