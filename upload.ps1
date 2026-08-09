# ---------------------------------------------------------------------------
# GitHub 업로드 — 실행 콘솔(athena.py)의 [5] 올리기 에서 실행됩니다.
#
# 올릴 때마다 자동으로:
#   1. 버전이 하나 올라갑니다            (VERSION 파일, 예: 2.2.0 → 2.2.1)
#   2. 무엇이 바뀌었는지 요약합니다      (커밋 메시지 + CHANGELOG.md 에 기록)
#   3. 커밋 → 버전 태그(v2.2.1) → push 까지 한 번에 처리합니다
#
# 버전을 직접 정하고 싶으면:
#   - 올리기 직전 물어볼 때 원하는 버전(예: 2.3.0)을 입력하거나
#   - VERSION 파일을 미리 고쳐 두면 됩니다.
#
# -NoPause : 끝나고 엔터를 기다리지 않습니다 (콘솔이 알아서 멈춰 줍니다).
# ---------------------------------------------------------------------------

param([switch]$NoPause)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location -LiteralPath $PSScriptRoot

$REPO_URL     = 'https://github.com/Heptagon372/athena-signal'
$LINE         = '============================================================'
$VERSION_FILE = Join-Path $PSScriptRoot 'VERSION'
$CHANGELOG    = Join-Path $PSScriptRoot 'CHANGELOG.md'

function Stop-Here {
    Write-Host ''
    if (-not $NoPause) { Read-Host '엔터를 누르면 창이 닫힙니다' }
    exit
}

# ---------------------------------------------------------------------------
# 버전 계산
# ---------------------------------------------------------------------------
function Get-CurrentVersion {
    if (Test-Path $VERSION_FILE) {
        $v = (Get-Content $VERSION_FILE -Raw).Trim()
        if ($v -match '^v?\d+(\.\d+){0,2}$') { return $v.TrimStart('v') }
    }
    return '2.2.0'
}

function ConvertTo-ThreePart([string]$v) {
    $p = @($v.Split('.'))
    while ($p.Count -lt 3) { $p += '0' }
    return ($p[0..2] -join '.')
}

function Get-AutoVersion {
    # 아직 태그로 올린 적 없는 버전이면 그대로 쓰고,
    # 이미 올렸던 버전이면 끝자리를 하나씩 올립니다.
    $cand = ConvertTo-ThreePart (Get-CurrentVersion)
    while (& git tag -l "v$cand") {
        $p = $cand.Split('.')
        $cand = '{0}.{1}.{2}' -f $p[0], $p[1], ([int]$p[2] + 1)
    }
    return $cand
}

Write-Host $LINE
Write-Host '  GitHub 업로드'
Write-Host "  $REPO_URL"
Write-Host $LINE
Write-Host ''

# ---------------------------------------------------------------------------
# [1/4] 변경된 파일 확인
# ---------------------------------------------------------------------------
Write-Host '[1/4] 변경된 파일을 확인합니다...'
Write-Host ''

$changes = @(& git -c core.quotepath=false status --porcelain)

if (-not $changes) {
    # 지난번에 커밋까지 됐는데 업로드만 실패한 경우를 잡아 줍니다.
    $ahead = & git rev-list --count "@{u}..HEAD" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ahead -and [int]$ahead -gt 0) {
        Write-Host "  변경된 파일은 없지만, 아직 안 올라간 커밋이 $ahead 개 있습니다."
        Write-Host '  이어서 업로드합니다...'
        & git push
        if ($LASTEXITCODE -eq 0) {
            & git push origin --tags
            Write-Host ''
            Write-Host '  완료되었습니다.'
        } else {
            Write-Host ''
            Write-Host '  업로드에 실패했습니다. 인터넷 연결을 확인하고 다시 해 보세요.'
        }
    } else {
        Write-Host '  변경된 파일이 없습니다. 올릴 것이 없습니다.'
    }
    Stop-Here
}

& git -c core.quotepath=false status --short
Write-Host ''

# ---------------------------------------------------------------------------
# 비밀 정보로 보이는 파일이 섞여 있는지 검사
# (.gitignore 를 통과한 새 파일 중 위험해 보이는 이름을 잡아냅니다)
# ---------------------------------------------------------------------------
$pattern = '(?i)(api_key|secret|token|password|\.env|\.pem|\.key$|\.pfx)'
$risky = & git add -A --dry-run | Where-Object { $_ -match $pattern }

if ($risky) {
    Write-Host '------------------------------------------------------------'
    Write-Host '  [경고] 비밀 정보로 보이는 파일이 포함되어 있습니다:'
    Write-Host '------------------------------------------------------------'
    $risky | ForEach-Object { Write-Host "  $_" }
    Write-Host '------------------------------------------------------------'
    Write-Host '  이 저장소는 공개(public) 입니다. 올리면 누구나 볼 수 있습니다.'
    Write-Host '  확실하지 않으면 지금 중단하고 .gitignore 에 추가하세요.'
    Write-Host ''
    $go = Read-Host '  그래도 진행하려면 yes 를 입력하세요'
    if ($go -ne 'yes') {
        Write-Host ''
        Write-Host '  중단했습니다. 아무것도 올라가지 않았습니다.'
        Stop-Here
    }
    Write-Host ''
}

# ---------------------------------------------------------------------------
# [2/4] 변경 내용 요약 만들기
# ---------------------------------------------------------------------------
$counts = [ordered]@{ '수정' = 0; '추가' = 0; '삭제' = 0; '이름변경' = 0 }
$areas  = @{}

foreach ($line in $changes) {
    if ($line.Length -lt 4) { continue }
    $code = $line.Substring(0, 2)
    $path = $line.Substring(3)
    if ($path -match ' -> ') { $path = ($path -split ' -> ')[-1] }
    $path = $path.Trim('"')

    if     ($code.Contains('?')) { $kind = '추가' }
    elseif ($code.Contains('A')) { $kind = '추가' }
    elseif ($code.Contains('D')) { $kind = '삭제' }
    elseif ($code.Contains('R')) { $kind = '이름변경' }
    else                         { $kind = '수정' }
    $counts[$kind] = $counts[$kind] + 1

    if     ($path.Contains('/')) { $area = $path.Split('/')[0] }
    elseif ($path -like '*.md')  { $area = '문서' }
    else                         { $area = '루트' }

    if (-not $areas.ContainsKey($area)) { $areas[$area] = @{} }
    if (-not $areas[$area].ContainsKey($kind)) { $areas[$area][$kind] = @() }
    # 새 폴더는 "cli/" 처럼 경로가 / 로 끝나므로 폴더명을 그대로 살립니다.
    $name = [IO.Path]::GetFileName($path.TrimEnd('/'))
    if ($path.EndsWith('/')) { $name = $name + '/' }
    $areas[$area][$kind] += $name
}

$countParts = @()
foreach ($k in @('수정', '추가', '삭제', '이름변경')) {
    if ($counts[$k] -gt 0) { $countParts += "$k $($counts[$k])" }
}
$countLine = $countParts -join ' · '

$bodyLines = @()
foreach ($area in ($areas.Keys | Sort-Object)) {
    $frags = @()
    foreach ($k in @('수정', '추가', '삭제', '이름변경')) {
        if ($areas[$area].ContainsKey($k)) {
            $files = @($areas[$area][$k])
            if ($files.Count -le 3) {
                $names = $files -join ', '
            } else {
                $names = ($files[0..1] -join ', ') + " 외 $($files.Count - 2)개"
            }
            $frags += "$names $k"
        }
    }
    $bodyLines += ('- ' + $area + ': ' + ($frags -join ' · '))
}

Write-Host '[2/4] 변경 내용 요약'
Write-Host ''
Write-Host "  $countLine"
foreach ($b in $bodyLines) { Write-Host "  $b" }
Write-Host ''

# ---------------------------------------------------------------------------
# [3/4] 버전 정하기
# ---------------------------------------------------------------------------
$auto = Get-AutoVersion
$inp  = Read-Host "[3/4] 엔터=v$auto 로 올리기 · 다른 버전은 직접 입력(예: 2.3.0) · q=중단"

if ($inp -eq 'q') {
    Write-Host ''
    Write-Host '  중단했습니다. 아무것도 올라가지 않았습니다.'
    Stop-Here
}
if ([string]::IsNullOrWhiteSpace($inp)) {
    $ver = $auto
} elseif ($inp -match '^v?\d+(\.\d+){0,2}$') {
    $ver = ConvertTo-ThreePart ($inp.TrimStart('v'))
} else {
    Write-Host "  '$inp' 는 버전 형식이 아니어서 v$auto 를 사용합니다."
    $ver = $auto
}

# ---------------------------------------------------------------------------
# [4/4] 커밋하고 업로드
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host "[4/4] v$ver 로 커밋하고 업로드합니다..."

# VERSION / CHANGELOG.md 갱신 — 이번 커밋에 함께 들어갑니다.
Set-Content -LiteralPath $VERSION_FILE -Value $ver -Encoding ASCII

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
$entryLines = @("## v$ver - $stamp", '', $countLine) + $bodyLines + @('')
$entry = $entryLines -join "`r`n"
$title = '# 변경 이력'
if (Test-Path $CHANGELOG) {
    $old = (Get-Content $CHANGELOG -Raw) -replace '^# 변경 이력\s*\r?\n', ''
    $new = "$title`r`n`r`n$entry`r`n$($old.TrimStart())"
} else {
    $new = "$title`r`n`r`n$entry"
}
Set-Content -LiteralPath $CHANGELOG -Value $new -Encoding UTF8

& git add -A
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  파일을 준비하지 못했습니다. 위 메시지를 확인하세요.'
    Stop-Here
}

$subject = "v$ver | $countLine"
& git commit -m $subject -m ($bodyLines -join "`n")
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  커밋에 실패했습니다. 위 메시지를 확인하세요.'
    Stop-Here
}

# 버전 태그 — GitHub 의 Tags / Releases 목록에서 버전별로 볼 수 있게 됩니다.
$tagExists = & git tag -l "v$ver"
if (-not $tagExists) {
    & git tag "v$ver"
} else {
    Write-Host "  (v$ver 태그가 이미 있어 새로 만들지는 않았습니다.)"
}

& git push
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  업로드에 실패했습니다.'
    Write-Host '  커밋은 이미 저장되어 있으니, 인터넷 연결을 확인한 뒤'
    Write-Host '  [5] 올리기 를 다시 하면 그대로 올라갑니다.'
    Stop-Here
}

& git push origin "v$ver"
if ($LASTEXITCODE -ne 0) {
    Write-Host '  (버전 태그 업로드는 실패했지만, 코드는 올라갔습니다. 다음 올리기 때 다시 시도합니다.)'
}

Write-Host ''
Write-Host $LINE
Write-Host "  완료되었습니다 — athena-signal v$ver"
Write-Host "  $REPO_URL"
Write-Host "  버전 목록: $REPO_URL/tags"
Write-Host $LINE
Stop-Here
