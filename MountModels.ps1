# MountModels.ps1
# This script creates and mounts the virtual disk to WSL. It requires Administrator privileges.

if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting Administrator privileges..."
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

$vhdxPath = "F:\qwen_models.vhdx"

# Create the VHDX if it doesn't exist yet
if (-not (Test-Path $vhdxPath)) {
    Write-Host "Creating 50GB virtual disk at $vhdxPath..."
    New-VHD -Path $vhdxPath -SizeBytes 50GB -Dynamic | Out-Null
    Write-Host "Virtual disk created!"
}

Write-Host "Mounting VHDX to WSL..."
# Unmount first in case it's in a bad state
wsl --unmount $vhdxPath 2>$null
# Mount the drive with a specific name
wsl --mount $vhdxPath --vhd --name qwen_models

Write-Host "Disk mounted successfully! You can now start the vLLM server."
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
