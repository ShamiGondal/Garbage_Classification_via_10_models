# Copy cuDNN files to CUDA 13.1 directory
# Run this script as Administrator

Write-Host "Copying cuDNN 9.17 files to CUDA v13.1 directory..." -ForegroundColor Green

$cudnn_base = "C:\Program Files\NVIDIA\CUDNN\v9.17"
$cuda_base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

# Check if source exists
if (-not (Test-Path $cudnn_base)) {
    Write-Host "Error: cuDNN not found at $cudnn_base" -ForegroundColor Red
    exit 1
}

# Check if destination exists
if (-not (Test-Path $cuda_base)) {
    Write-Host "Error: CUDA not found at $cuda_base" -ForegroundColor Red
    exit 1
}

# Copy bin files
Write-Host "`nCopying bin files..." -ForegroundColor Yellow
$bin_src = Join-Path $cudnn_base "bin\13.1"
$bin_dst = Join-Path $cuda_base "bin"
if (Test-Path $bin_src) {
    Copy-Item "$bin_src\*" -Destination $bin_dst -Force
    Write-Host "  ✓ Bin files copied" -ForegroundColor Green
} else {
    Write-Host "  ✗ Bin source not found" -ForegroundColor Red
}

# Copy include files
Write-Host "`nCopying include files..." -ForegroundColor Yellow
$include_src = Join-Path $cudnn_base "include\13.1"
$include_dst = Join-Path $cuda_base "include"
if (Test-Path $include_src) {
    Copy-Item "$include_src\*" -Destination $include_dst -Force -Recurse
    Write-Host "  ✓ Include files copied" -ForegroundColor Green
} else {
    Write-Host "  ✗ Include source not found" -ForegroundColor Red
}

# Copy lib files
Write-Host "`nCopying lib files..." -ForegroundColor Yellow
$lib_src = Join-Path $cudnn_base "lib\13.1"
$lib_dst = Join-Path $cuda_base "lib"
if (Test-Path $lib_src) {
    Copy-Item "$lib_src\*" -Destination $lib_dst -Force -Recurse
    Write-Host "  ✓ Lib files copied" -ForegroundColor Green
} else {
    Write-Host "  ✗ Lib source not found" -ForegroundColor Red
}

Write-Host "`n✅ Copy complete!" -ForegroundColor Green
Write-Host "`nVerifying files..." -ForegroundColor Yellow

# Verify
$dll_count = (Get-ChildItem "$cuda_base\bin\cudnn*.dll" -ErrorAction SilentlyContinue).Count
if ($dll_count -gt 0) {
    Write-Host "  ✓ Found $dll_count cuDNN DLL files in CUDA bin directory" -ForegroundColor Green
} else {
    Write-Host "  ✗ No cuDNN DLL files found" -ForegroundColor Red
}

Write-Host "`n⚠️  IMPORTANT: Restart your computer or at least restart the terminal!" -ForegroundColor Cyan
Write-Host "   Then run: python test_gpu.py" -ForegroundColor Cyan

