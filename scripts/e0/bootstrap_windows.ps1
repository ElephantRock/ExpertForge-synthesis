$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

$expectedBranch = 'e0/qualification-bootstrap'
$branch = (git branch --show-current).Trim()
if ($branch -ne $expectedBranch) {
    throw "Expected branch '$expectedBranch' but found '$branch'."
}

$origin = (git remote get-url origin).Trim()
if ($origin -notmatch 'ElephantRock/ExpertForge-synthesis') {
    throw "Unexpected origin: $origin"
}

$commit = (git rev-parse HEAD).Trim()
$dirty = -not [string]::IsNullOrWhiteSpace((git status --porcelain | Out-String).Trim())

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = 'py -3'
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = 'python'
} else {
    throw 'Python 3 was not found on PATH.'
}

if (-not (Test-Path '.venv')) {
    if ($python -eq 'py -3') { & py -3 -m venv .venv }
    else { & python -m venv .venv }
}

$venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$pythonVersion = (& $venvPython --version 2>&1 | Out-String).Trim()

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$disk = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($RepoRoot).Substring(0,1))

$gpuInfo = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $rows = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
    foreach ($row in $rows) {
        $parts = $row -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 3) {
            $gpuInfo += [ordered]@{
                name = $parts[0]
                memory_mib = [int64]$parts[1]
                driver_version = $parts[2]
            }
        }
    }
}

$torch = [ordered]@{
    installed = $false
    version = $null
    cuda_available = $false
    cuda_runtime = $null
    gpu_names = @()
}
try {
    $torchJson = & $venvPython -c "import json,torch; print(json.dumps({'installed':True,'version':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $torchJson) {
        $parsed = $torchJson | ConvertFrom-Json
        $torch.installed = $parsed.installed
        $torch.version = $parsed.version
        $torch.cuda_available = $parsed.cuda_available
        $torch.cuda_runtime = $parsed.cuda_runtime
        $torch.gpu_names = @($parsed.gpu_names)
    }
} catch {}

$snapshot = [ordered]@{
    schema_id = 'E0-WINDOWS-ENVIRONMENT-SNAPSHOT-v0'
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    repository = [ordered]@{
        origin = $origin
        branch = $branch
        commit_sha = $commit
        dirty_at_capture = $dirty
    }
    os = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build_number = $os.BuildNumber
        architecture = $os.OSArchitecture
    }
    cpu = [ordered]@{
        name = $cpu.Name.Trim()
        logical_processors = [int]$computer.NumberOfLogicalProcessors
    }
    host_memory_bytes = [int64]$computer.TotalPhysicalMemory
    workspace = [ordered]@{
        repo_root = '<local-repository>'
        drive = $disk.Name
        free_bytes = [int64]$disk.Free
        used_bytes = [int64]$disk.Used
    }
    python = [ordered]@{
        version = $pythonVersion
        virtual_environment = '.venv'
    }
    nvidia_gpus = $gpuInfo
    pytorch = $torch
}

$outDir = Join-Path $RepoRoot 'docs\experiments\e0'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir 'windows_environment.snapshot.json'
$snapshot | ConvertTo-Json -Depth 8 | Set-Content -Path $outFile -Encoding UTF8

$statusFile = Join-Path $outDir 'LOCAL_BOOTSTRAP_STATUS.md'
@"
# E0 Windows Local Bootstrap Status

- Git commit at capture: ``$commit``
- Branch: ``$branch``
- Python: ``$pythonVersion``
- Q1 local reproduction: **PENDING**
- Q2 M0 qualification: **BLOCKED ON Q1 LOCAL REPRODUCTION**
- Q3 S0 qualification: **BLOCKED ON Q1 LOCAL REPRODUCTION**

Next action: reproduce Q1 locally, commit the environment snapshot + Q1 audit, push, and report the commit SHA in GitHub issue #1.
"@ | Set-Content -Path $statusFile -Encoding UTF8

Write-Host "Wrote: $outFile"
Write-Host "Wrote: $statusFile"
Write-Host 'Do not start Q2/Q3 until Q1 has reproduced and this first deliverable is pushed to GitHub.'
