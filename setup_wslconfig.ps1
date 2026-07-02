# Run this from a normal (non-admin) Windows PowerShell BEFORE setup_wham.sh.
# WHAM needs more RAM than WSL2's default 50%-of-host allocation gives it
# (it OOM-kills partway through inference otherwise). This writes a generous
# .wslconfig and restarts WSL to apply it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup_wslconfig.ps1

$totalGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "Detected $totalGB GB of system RAM."

if ($totalGB -lt 16) {
    Write-Warning "Less than 16 GB RAM detected. WHAM may still OOM even with this config."
}

# Leave ~4GB for Windows itself; cap WSL memory below that.
$wslMemGB = [math]::Max(6, $totalGB - 4)
$swapGB = 16

$config = @"
[wsl2]
memory=${wslMemGB}GB
swap=${swapGB}GB
processors=$([Environment]::ProcessorCount)
"@

$path = "$env:USERPROFILE\.wslconfig"
if (Test-Path $path) {
    Write-Host "Backing up existing .wslconfig to .wslconfig.bak"
    Copy-Item $path "$path.bak" -Force
}
Set-Content -Path $path -Value $config -Encoding ascii
Write-Host "Wrote $path :"
Write-Host $config

Write-Host "`nRestarting WSL to apply..."
wsl --shutdown
Start-Sleep -Seconds 2
Write-Host "Done. Run setup_wham.sh next (inside WSL)."
