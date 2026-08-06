# ---------------------------------------------------------------------------
# GitHub 업로드 — 업로드.bat 에서 실행됩니다.
# 변경 파일 확인 → 커밋 → push 를 한 번에 처리합니다.
# ---------------------------------------------------------------------------

[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location -LiteralPath $PSScriptRoot

$REPO_URL = 'https://github.com/Heptagon372/athena-signal'
$LINE     = '============================================================'

function Stop-Here {
    Write-Host ''
    Read-Host '엔터를 누르면 창이 닫힙니다'
    exit
}

Write-Host $LINE
Write-Host '  GitHub 업로드'
Write-Host "  $REPO_URL"
Write-Host $LINE
Write-Host ''

# ---------------------------------------------------------------------------
# [1/3] 변경된 파일 확인
# ---------------------------------------------------------------------------
Write-Host '[1/3] 변경된 파일을 확인합니다...'
Write-Host ''

$changes = & git status --porcelain
if (-not $changes) {
    Write-Host '  변경된 파일이 없습니다. 올릴 것이 없습니다.'
    Stop-Here
}

& git status --short
Write-Host ''

# ---------------------------------------------------------------------------
# 비밀 정보로 보이는 파일이 섞여 있는지 검사
# (.gitignore 를 통과한 새 파일 중 위험해 보이는 이름을 잡아냅니다)
# ---------------------------------------------------------------------------
$pattern = '(?i)(api_key|secret|token|password|credential|\.env|\.pem|\.key$|\.pfx)'
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
# [2/3] 커밋
# ---------------------------------------------------------------------------
$msg = Read-Host '커밋 메시지 (엔터=자동)'
if ([string]::IsNullOrWhiteSpace($msg)) {
    $msg = '작업 내용 저장 ' + (Get-Date -Format 'yyyy-MM-dd HH:mm')
}

Write-Host ''
Write-Host '[2/3] 커밋합니다...'

& git add -A
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  파일을 준비하지 못했습니다. 위 메시지를 확인하세요.'
    Stop-Here
}

& git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  커밋에 실패했습니다. 위 메시지를 확인하세요.'
    Stop-Here
}

# ---------------------------------------------------------------------------
# [3/3] 업로드
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '[3/3] GitHub 에 업로드합니다...'

& git push
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  업로드에 실패했습니다.'
    Write-Host '  커밋은 이미 저장되어 있으니, 인터넷 연결을 확인한 뒤'
    Write-Host '  업로드.bat 을 다시 실행하면 그대로 올라갑니다.'
    Stop-Here
}

Write-Host ''
Write-Host $LINE
Write-Host '  완료되었습니다.'
Write-Host "  $REPO_URL"
Write-Host $LINE
Stop-Here
