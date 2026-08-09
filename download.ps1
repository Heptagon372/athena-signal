# ---------------------------------------------------------------------------
# GitHub 가져오기 — 실행 콘솔(athena.py)의 [6] 받아오기 에서 실행됩니다.
# 새 내용 확인 → 내려받기 → 새 라이브러리 설치를 한 번에 처리합니다.
#
# [5] 올리기 의 반대 방향입니다. 다른 PC 에서 올린 최신 코드를 이 PC 로
# 받아옵니다. 로컬에 저장 안 된 변경이 있으면 먼저 물어보고, 덮어쓰지
# 않습니다.
#
# -NoPause : 끝나고 엔터를 기다리지 않습니다 (콘솔이 알아서 멈춰 줍니다).
# ---------------------------------------------------------------------------

param([switch]$NoPause)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location -LiteralPath $PSScriptRoot

$REPO_URL = 'https://github.com/Heptagon372/athena-signal'
$LINE     = '============================================================'

function Stop-Here {
    Write-Host ''
    if (-not $NoPause) { Read-Host '엔터를 누르면 창이 닫힙니다' }
    exit
}

Write-Host $LINE
Write-Host '  GitHub 가져오기'
Write-Host "  $REPO_URL"
Write-Host $LINE
Write-Host ''

# 현재 브랜치와 그 짝이 되는 원격 브랜치를 찾습니다.
$branch = & git rev-parse --abbrev-ref HEAD
if ($LASTEXITCODE -ne 0) {
    Write-Host '  git 저장소가 아닙니다. 폴더 위치를 확인하세요.'
    Stop-Here
}
$remote = "origin/$branch"

# ---------------------------------------------------------------------------
# [1/3] 새 내용 확인
# ---------------------------------------------------------------------------
Write-Host "[1/3] GitHub 에서 새 내용을 확인합니다... (브랜치: $branch)"
Write-Host ''

& git fetch origin
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '  확인에 실패했습니다. 인터넷 연결을 확인하세요.'
    Write-Host '  아무것도 바뀌지 않았습니다.'
    Stop-Here
}

$local  = & git rev-parse HEAD
$origin = & git rev-parse $remote
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host "  GitHub 에 $remote 브랜치가 없습니다."
    Stop-Here
}

if ($local -eq $origin) {
    Write-Host ''
    Write-Host '  이미 최신입니다. 받을 것이 없습니다.'
    Stop-Here
}

# 로컬이 원격보다 앞서 있는지(= 아직 안 올린 커밋이 있는지) 확인합니다.
& git merge-base --is-ancestor HEAD $remote
$canFastForward = ($LASTEXITCODE -eq 0)

if (-not $canFastForward) {
    Write-Host ''
    Write-Host '------------------------------------------------------------'
    Write-Host '  [멈춤] 이 PC 에만 있는 커밋이 있습니다.'
    Write-Host '------------------------------------------------------------'
    & git log --oneline "$remote..HEAD"
    Write-Host '------------------------------------------------------------'
    Write-Host '  그냥 받아오면 위 작업이 묻힐 수 있어 멈췄습니다.'
    Write-Host '  먼저 [5] 올리기 로 이 PC 의 작업을 올린 뒤,'
    Write-Host '  다시 [6] 받아오기 를 실행하세요.'
    Stop-Here
}

# 무엇이 바뀌는지 보여줍니다.
Write-Host ''
Write-Host '  새로 올라온 작업:'
& git log --oneline "HEAD..$remote"
Write-Host ''
Write-Host '  바뀌는 파일:'
& git diff --stat "HEAD..$remote"
Write-Host ''

# ---------------------------------------------------------------------------
# 저장 안 된 변경이 있으면 먼저 처리합니다 (덮어쓰지 않습니다)
# ---------------------------------------------------------------------------
$stashed = $false
$changes = & git status --porcelain

if ($changes) {
    Write-Host '------------------------------------------------------------'
    Write-Host '  [주의] 이 PC 에 저장(커밋) 안 된 변경이 있습니다:'
    Write-Host '------------------------------------------------------------'
    & git status --short
    Write-Host '------------------------------------------------------------'

    # 내가 고친 파일이 GitHub 쪽에서도 바뀌었으면 되돌릴 때 충돌이 납니다.
    # 미리 알려줍니다.
    $mine     = (& git diff --name-only HEAD) + (& git ls-files --others --exclude-standard)
    $upstream = & git diff --name-only "HEAD..$remote"
    $clash    = $mine | Where-Object { $upstream -contains $_ }

    if ($clash) {
        Write-Host '  [경고] 아래 파일은 이 PC 와 GitHub 양쪽에서 바뀌었습니다:'
        $clash | ForEach-Object { Write-Host "      $_" }
        Write-Host '  되돌릴 때 충돌이 날 수 있습니다. 그래도 작업은 사라지지 않습니다.'
        Write-Host '  안전하게 하려면 중단하고 [5] 올리기 를 먼저 실행하세요.'
        Write-Host '------------------------------------------------------------'
    }

    Write-Host '  yes 를 입력하면 이 변경을 잠시 치워두고 받아온 뒤,'
    Write-Host '  다시 원래대로 되돌려 놓습니다. (사라지지 않습니다)'
    Write-Host '  그냥 두고 싶으면 중단하고 [5] 올리기 를 먼저 실행하세요.'
    Write-Host ''
    $go = Read-Host '  계속하려면 yes 를 입력하세요'
    if ($go -ne 'yes') {
        Write-Host ''
        Write-Host '  중단했습니다. 아무것도 바뀌지 않았습니다.'
        Stop-Here
    }

    Write-Host ''
    Write-Host '  변경을 잠시 보관합니다...'
    $label = '가져오기 전 임시 보관 ' + (Get-Date -Format 'yyyy-MM-dd HH:mm')
    & git stash push -u -m $label
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host '  보관에 실패했습니다. 아무것도 바뀌지 않았습니다.'
        Stop-Here
    }
    $stashed = $true
    Write-Host ''
}

# ---------------------------------------------------------------------------
# [2/3] 내려받기
# ---------------------------------------------------------------------------
Write-Host '[2/3] 최신 코드를 받아옵니다...'
Write-Host ''

& git merge --ff-only $remote
$mergeFailed = ($LASTEXITCODE -ne 0)

if ($stashed) {
    Write-Host ''
    Write-Host '  보관해 둔 변경을 되돌립니다...'
    & git stash pop
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host '------------------------------------------------------------'
        Write-Host '  [주의] 되돌리는 중 충돌이 났습니다.'
        Write-Host ''
        Write-Host '  작업은 사라지지 않았습니다. 아래 파일이 충돌했습니다:'
        & git diff --name-only --diff-filter=U | ForEach-Object { Write-Host "      $_" }
        Write-Host ''
        Write-Host '  이 파일들을 열어 보면 이렇게 두 버전이 같이 들어 있습니다:'
        Write-Host '      <<<<<<< Updated upstream'
        Write-Host '          (GitHub 에서 받아온 내용)'
        Write-Host '      ======='
        Write-Host '          (이 PC 에서 고친 내용)'
        Write-Host '      >>>>>>> Stashed changes'
        Write-Host ''
        Write-Host '  남길 쪽만 두고 <<<<<<< ======= >>>>>>> 줄은 지우세요.'
        Write-Host '  정리가 끝나면 아래를 실행해 보관본을 지우면 됩니다:'
        Write-Host '      git stash drop'
        Write-Host '------------------------------------------------------------'
        Stop-Here
    }
}

if ($mergeFailed) {
    Write-Host ''
    Write-Host '  받아오기에 실패했습니다. 위 메시지를 확인하세요.'
    Stop-Here
}

# ---------------------------------------------------------------------------
# [3/3] 새 라이브러리 설치
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '[3/3] 새로 필요한 라이브러리를 확인합니다...'

$reqChanged = & git diff --name-only $local HEAD -- requirements.txt

if ($reqChanged) {
    Write-Host '  requirements.txt 가 바뀌었습니다. 설치합니다...'
    Write-Host ''
    & python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host '  라이브러리 설치에 실패했습니다.'
        Write-Host '  코드는 이미 최신이니, 아래를 직접 실행해 보세요:'
        Write-Host '      python -m pip install -r requirements.txt'
        Stop-Here
    }
} else {
    Write-Host '  새로 설치할 것이 없습니다.'
}

Write-Host ''
Write-Host $LINE
Write-Host '  완료되었습니다.'
Write-Host '  서버가 켜져 있었다면 [3] 업데이트 로 다시 시작하세요.'
Write-Host $LINE
Stop-Here
