$ErrorActionPreference = "Stop"

$package = if ($env:IVONAR_PACKAGE) { $env:IVONAR_PACKAGE } else { "https://github.com/LuisCode28/ivonar-inference/archive/refs/heads/main.tar.gz" }
$backend = if ($env:IVONAR_TORCH_BACKEND) { $env:IVONAR_TORCH_BACKEND } else { "auto" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv, which provides Python for Ivonar"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Ivonar"
uv tool install --force --python 3.12 --torch-backend $backend "ivonar-inference @ $package"
if ($LASTEXITCODE -ne 0) { throw "Installing Ivonar failed. If uv is more than a year old, run 'uv self update' and try again." }

$bin = (uv tool dir --bin).Trim()
$normal = { param($path) $path.Replace("/", "\").TrimEnd("\").ToLowerInvariant() }
$onPath = @($env:Path -split ";" | Where-Object { $_ } | ForEach-Object { & $normal $_ }) -contains (& $normal $bin)
if (-not $onPath) { uv tool update-shell | Out-Null }

Write-Host "Starting Ivonar; next time run: ivonar serve"
& (Join-Path $bin "ivonar.exe") serve @args
