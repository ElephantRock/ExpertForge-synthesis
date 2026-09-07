$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

$venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'Missing .venv. Run scripts/e0/bootstrap_windows.ps1 first.'
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r .\requirements\e0-q1.txt

$out = Join-Path $RepoRoot 'local_data\e0_qualification_bootstrap\Q1'
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $venvPython .\scripts\e0\q1_cmdr_bootstrap.py --out $out
& $venvPython .\scripts\e0\q1_verify_and_audit.py --root $out

$expected = [ordered]@{
    'train_ID.jsonl' = '79daa2c007bc914e228a0312d8278b8f5ca7cb2c28f1450d508f9924362aacf1'
    'eval_ID.jsonl' = 'b66b6629173449f71022fba0e7907826733aa7ae28e0b7502716ba4fdd45fce8'
    'eval_STRUCT.jsonl' = '688865d27b8e1fd208b2b453670306d1b4ecc9ff9a21130a297e67cc4b07c7d8'
    'CMDR-Lex-v1.json' = 'c3d44e7a99163456ef149fdd1b1caf2fd694e8412e6f082480cb05c2ef237471'
}

$hashResults = [ordered]@{}
$allHashesMatch = $true
foreach ($name in $expected.Keys) {
    $path = Join-Path $out $name
    $actual = (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLowerInvariant()
    $match = ($actual -eq $expected[$name])
    $hashResults[$name] = [ordered]@{ expected = $expected[$name]; actual = $actual; match = $match }
    if (-not $match) { $allHashesMatch = $false }
}

$proof = Get-Content (Join-Path $out 'proof_depth_audit.json') -Raw | ConvertFrom-Json
$counterfactual = Get-Content (Join-Path $out 'counterfactual_audit.json') -Raw | ConvertFrom-Json
$shortcut = Get-Content (Join-Path $out 'shortcut_audit.json') -Raw | ConvertFrom-Json
$manifest = Get-Content (Join-Path $out 'CMDR-QUAL-v1.manifest.json') -Raw | ConvertFrom-Json

$pass = $allHashesMatch -and
    ($proof.status -eq 'PASS') -and
    ($counterfactual.status -eq 'PASS') -and
    ($shortcut.status -eq 'PASS') -and
    ($counterfactual.families -eq 12000) -and
    ($counterfactual.exact_unigram_invariance_families -eq 12000) -and
    ($counterfactual.exact_bigram_invariance_families -eq 12000) -and
    ($manifest.maximum_student_token_count_with_DECIDE -eq 194)

$summary = [ordered]@{
    schema_id = 'E0-Q1-LOCAL-REPRODUCTION-v0'
    status = $(if ($pass) { 'PASS' } else { 'FAIL' })
    git_commit = (git rev-parse HEAD).Trim()
    qualification_examples = 36000
    qualification_families = 12000
    max_student_tokens_with_DECIDE = $manifest.maximum_student_token_count_with_DECIDE
    proof_depth_status = $proof.status
    counterfactual_status = $counterfactual.status
    shortcut_status = $shortcut.status
    shortcut_gate = 0.38
    expected_shortcut_score = 0.3333333333333333
    corpus_hashes = $hashResults
}

$docs = Join-Path $RepoRoot 'docs\experiments\e0'
New-Item -ItemType Directory -Force -Path $docs | Out-Null
$summaryPath = Join-Path $docs 'q1_local_reproduction.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

$statusPath = Join-Path $docs 'LOCAL_BOOTSTRAP_STATUS.md'
$state = if ($pass) { '**PASS**' } else { '**FAIL — STOP**' }
@"
# E0 Windows Local Bootstrap Status

- Git commit at Q1 reproduction: ``$((git rev-parse HEAD).Trim())``
- Q1 local reproduction: $state
- Q2 M0 qualification: **$(if ($pass) { 'READY AFTER PUSH + ISSUE REPORT' } else { 'BLOCKED' })**
- Q3 S0 qualification: **$(if ($pass) { 'READY AFTER PUSH + ISSUE REPORT' } else { 'BLOCKED' })**

Generated Q1 data remains under ``local_data/`` and is intentionally not committed. Commit ``windows_environment.snapshot.json`` and ``q1_local_reproduction.json`` instead.
"@ | Set-Content -Path $statusPath -Encoding UTF8

Write-Host "Q1 reproduction status: $($summary.status)"
Write-Host "Summary: $summaryPath"
if (-not $pass) {
    throw 'Q1 did not reproduce exactly. Do not proceed to Q2/Q3; report the mismatch in GitHub issue #1.'
}
