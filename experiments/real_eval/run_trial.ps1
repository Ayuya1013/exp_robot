# run_trial.ps1 - 実機評価を 1 回行い、結果を results.csv に記録する
#   .\run_trial.ps1             : モデル選択 → ホーム姿勢 → 1 回実行 → 結果 / 時間 / 備考を入力
#   .\run_trial.ps1 -SaveHome   : ホーム姿勢を保存し直す(手で腕を動かして Enter)
#   .\run_trial.ps1 -TestSheet  : スプレッドシートへの接続だけ確かめる(行は書かない)。未送信の行があれば送る
# 実行条件(ポート・カメラ・時間・タスク文)は sim\scripts\record.ps1 の -Policy と同じ。
# 結果は results.csv(控え)と Google スプレッドシート(sheets_config.json の URL)の両方に書く。

param(
    [switch]$SaveHome,
    [switch]$TestSheet
)

# ===== 設定(record.ps1 と同じ値) =====
$FollowerPort  = "COM4"
$LeaderPort    = "COM5"
$Task          = "Press the power button"
$EpisodeTime   = 20
$ResetTime     = 15
$WristRotation = 0
$Device        = "cpu"
$LerobotDir    = "C:\Users\yuya\lerobot"
$ModelsDir     = "C:\Users\yuya\lerobot\models"
# =====================================

$ErrorActionPreference = "Stop"
$Here       = $PSScriptRoot
$Python     = Join-Path $LerobotDir ".venv\Scripts\python.exe"
$RolloutExe = Join-Path $LerobotDir ".venv\Scripts\lerobot-rollout.exe"
$HomeScript = Join-Path $Here "go_home.py"
$HomePose   = Join-Path $Here "home_pose.json"
$RealLogDir = Join-Path $LerobotDir "logs\real"
$ResultsCsv = Join-Path $RealLogDir "results.csv"
$LogsDir    = Join-Path $RealLogDir "trials"
$LastModel  = Join-Path $RealLogDir ".last_model"
$SheetConf  = Join-Path $Here "sheets_config.json"
$Pending    = Join-Path $RealLogDir "pending_sheet.jsonl"   # スプレッドシートに送れなかった行
# 評価の録画(LeRobot データセット)は <HF_LEROBOT_HOME>\local\rollout\<モデル名>\rollout_eval_<試行ID>\ に置く
$LerobotHome = if ($env:HF_LEROBOT_HOME) { $env:HF_LEROBOT_HOME } else { Join-Path $env:USERPROFILE ".cache\huggingface\lerobot" }
$RolloutRoot = Join-Path $LerobotHome "local\rollout"

# venv を有効にしていなくても rerun.exe などが見つかるように
$env:PATH = (Join-Path $LerobotDir ".venv\Scripts") + ";" + $env:PATH
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }

# 外部コマンドを実行し、画面に出しながらログファイルにも書く。終了コードを返す
function Invoke-Logged {
    param([string]$Exe, [string[]]$ArgList, [string]$LogFile)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @ArgList 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Add-Content -Path $LogFile -Value $line -Encoding UTF8
        }
        return $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $prev }
}

# スプレッドシート(Apps Script ウェブアプリ)へ送る。成功で $true
function Send-Sheet {
    param([hashtable]$Body)
    if (-not (Test-Path $SheetConf)) { Write-Host "sheets_config.json がないため、スプレッドシートには送りません。" -ForegroundColor Yellow; return $false }
    $conf = Get-Content $SheetConf -Raw -Encoding UTF8 | ConvertFrom-Json
    $Body.token = $conf.token
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Depth 5 -Compress))
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $res = Invoke-RestMethod -Uri $conf.url -Method Post -Body $bytes -ContentType "application/json; charset=utf-8" -TimeoutSec 30
        if ($res.ok) { return $true }
        Write-Host "スプレッドシートがエラーを返しました: $($res.error)" -ForegroundColor Yellow
    }
    catch { Write-Host "スプレッドシートに送れませんでした: $($_.Exception.Message)" -ForegroundColor Yellow }
    return $false
}

# 前回までに送れなかった行を送り直す
function Send-Pending {
    if (-not (Test-Path $Pending)) { return }
    $left = @()
    foreach ($line in (Get-Content $Pending -Encoding UTF8 | Where-Object { $_.Trim() })) {
        $obj = $line | ConvertFrom-Json
        $row = [ordered]@{}
        foreach ($p in $obj.PSObject.Properties) { $row[$p.Name] = $p.Value }
        if (Send-Sheet @{ row = $row }) { Write-Host "未送信だった行を送りました: $($row.trial_id)" -ForegroundColor Green }
        else { $left += $line }
    }
    if ($left.Count -gt 0) { $left | Set-Content -Path $Pending -Encoding UTF8 }
    else { Remove-Item $Pending }
}

if ($TestSheet) {
    if (Send-Sheet @{ ping = $true }) { Write-Host "スプレッドシートに接続できました。" -ForegroundColor Green } else { exit 1 }
    Send-Pending
    exit 0
}

foreach ($f in @($Python, $RolloutExe, $HomeScript)) {
    if (-not (Test-Path $f)) { Write-Host "見つかりません: $f" -ForegroundColor Red; exit 1 }
}

# COM ポートの確認
$ports = [System.IO.Ports.SerialPort]::GetPortNames()
foreach ($p in @($FollowerPort, $LeaderPort)) {
    if ($ports -notcontains $p) {
        Write-Host "${p} が見つかりません。現在のポート: $($ports -join ', ')" -ForegroundColor Red
        exit 1
    }
}

# ---- ホーム姿勢の保存 ----
if ($SaveHome -or -not (Test-Path $HomePose)) {
    Write-Step "ホーム姿勢の保存"
    & $Python $HomeScript --port $FollowerPort --save
    if ($LASTEXITCODE -ne 0) { Write-Host "保存に失敗しました。" -ForegroundColor Red; exit 1 }
    if ($SaveHome) { exit 0 }
}

# ---- 1. モデル選択 ----
Write-Step "1. モデル選択"
$models = @(Get-ChildItem $ModelsDir -Directory | Where-Object {
    (Test-Path (Join-Path $_.FullName "config.json")) -and (Test-Path (Join-Path $_.FullName "model.safetensors"))
} | Sort-Object Name)
if ($models.Count -eq 0) { Write-Host "使えるモデルがありません: $ModelsDir" -ForegroundColor Red; exit 1 }

$past = @()
if (Test-Path $ResultsCsv) { $past = @(Import-Csv $ResultsCsv -Encoding UTF8) }
$last = if (Test-Path $LastModel) { (Get-Content $LastModel -Raw).Trim() } else { "" }

for ($i = 0; $i -lt $models.Count; $i++) {
    $m = $models[$i]
    $type = (Get-Content (Join-Path $m.FullName "config.json") -Raw | ConvertFrom-Json).type
    $rows = @($past | Where-Object { $_.model -eq $m.Name })
    $o  = @($rows | Where-Object { $_.result -eq "〇" }).Count
    $tr = @($rows | Where-Object { $_.result -eq "△" }).Count
    $x  = @($rows | Where-Object { $_.result -eq "×" }).Count
    $mark = if ($m.Name -eq $last) { "*" } else { " " }
    "{0}{1,3}) {2,-22} {3,-8} 実施 {4,2} 回 (〇{5} △{6} ×{7})" -f $mark, ($i + 1), $m.Name, $type, $rows.Count, $o, $tr, $x | Write-Host
}
if ($last) { Write-Host "  (* = 前回のモデル。Enter だけで同じモデル)" }

$model = $null
while (-not $model) {
    $ans = (Read-Host "番号を入力 (q で中止)").Trim()
    if ($ans -eq "q") { exit 0 }
    if ($ans -eq "" -and $last) { $model = $models | Where-Object { $_.Name -eq $last } | Select-Object -First 1 }
    elseif ($ans -match '^\d+$' -and [int]$ans -ge 1 -and [int]$ans -le $models.Count) { $model = $models[[int]$ans - 1] }
    if (-not $model) { Write-Host "1 ~ $($models.Count) で入力してください。" -ForegroundColor Yellow }
}
Set-Content -Path $LastModel -Value $model.Name -Encoding UTF8
$policyType = (Get-Content (Join-Path $model.FullName "config.json") -Raw | ConvertFrom-Json).type
$trialNo = @($past | Where-Object { $_.model -eq $model.Name }).Count + 1
Write-Host "選択: $($model.Name) ($policyType) / このモデルの $trialNo 回目" -ForegroundColor Green

# 試行 ID とログの置き場所
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$trialId = "${stamp}_$($model.Name)"
$trialDir = Join-Path $LogsDir $trialId
New-Item -ItemType Directory -Force $trialDir | Out-Null
$homeLog    = Join-Path $trialDir "home.log"
$rolloutLog = Join-Path $trialDir "rollout.log"

# ---- 2. ホーム姿勢 ----
Write-Step "2. ホーム姿勢へ移動"
Write-Host "腕の周りに物がないことを確認してください。"
Read-Host "Enter で移動開始" | Out-Null
$homeCode = Invoke-Logged -Exe $Python -ArgList @($HomeScript, "--port", $FollowerPort) -LogFile $homeLog
$homeLine = Select-String -Path $homeLog -Pattern "^HOME_RESULT " | Select-Object -Last 1
$homeErr = ""
if ($homeLine) { $homeErr = [math]::Round(($homeLine.Line.Substring(12) | ConvertFrom-Json).max_abs_err, 2) }
if ($homeCode -ne 0) {
    Write-Host "ホーム姿勢への移動で問題がありました(終了コード $homeCode)。" -ForegroundColor Yellow
    if ((Read-Host "このまま続けますか? [y/N]").Trim().ToLower() -ne "y") { exit 1 }
}

# ---- 3. 実行 ----
Write-Step "3. 実行($EpisodeTime 秒)"
Write-Host "エアコンが停止状態になっていることを確認してください。"
Read-Host "Enter で開始" | Out-Null

$datasetName = "rollout_eval_$trialId"
$datasetDir  = Join-Path (Join-Path $RolloutRoot $model.Name) $datasetName
New-Item -ItemType Directory -Force (Split-Path $datasetDir) | Out-Null   # モデルのフォルダ(録画のフォルダ自体は lerobot が作る)
$wrist = "wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG"
if ($WristRotation -ne 0) { $wrist += ", rotation: $WristRotation" }
$wrist += "}"
$top  = "top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}"
$cams = "{" + $wrist + ", " + $top + "}"

$rolloutArgs = @(
    "--robot.type=so101_follower",
    "--robot.port=$FollowerPort",
    "--robot.id=my_follower",
    "--robot.cameras=$cams",
    "--teleop.type=so101_leader",
    "--teleop.port=$LeaderPort",
    "--teleop.id=my_leader",
    "--display_data=true",
    "--display_ip=127.0.0.1",
    "--display_port=9876",
    "--dataset.repo_id=local/$datasetName",
    "--dataset.root=$datasetDir",
    "--dataset.no_stamp=true",
    "--dataset.single_task=$Task",
    "--dataset.num_episodes=1",
    "--dataset.episode_time_s=$EpisodeTime",
    "--dataset.reset_time_s=$ResetTime",
    "--dataset.push_to_hub=false",
    "--strategy.type=episodic",
    "--policy.path=$($model.FullName)",
    "--device=$Device"
)

$commit = (& git -C $LerobotDir rev-parse --short HEAD 2>$null)
$weights = Get-Item (Join-Path $model.FullName "model.safetensors")
@(
    "# trial_id : $trialId",
    "# model    : $($model.FullName) ($policyType)",
    "# weights  : $($weights.Length) bytes, $($weights.LastWriteTime.ToString('s'))",
    "# lerobot  : $commit",
    "# command  : $RolloutExe $($rolloutArgs -join ' ')",
    "# start    : $(Get-Date -Format s)",
    ""
) | Set-Content -Path $rolloutLog -Encoding UTF8

# Rerun の画面を先に別プロセスで起動しておく。lerobot-rollout に起動させると、
# 画面が出力のパイプを握ったままになり、画面を閉じるまでこのスクリプトが先へ進まない。
# (lerobot-rollout は、ポート 9876 で動いている画面があればそれにつなぐ)
if (-not (Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath (Join-Path $LerobotDir ".venv\Scripts\rerun.exe") -ArgumentList "--memory-limit", "10%" | Out-Null
    for ($i = 0; $i -lt 30 -and -not (Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue); $i++) {
        Start-Sleep -Milliseconds 500
    }
}

$t0 = Get-Date
$rolloutCode = Invoke-Logged -Exe $RolloutExe -ArgList $rolloutArgs -LogFile $rolloutLog
$t1 = Get-Date
$wall = [math]::Round(($t1 - $t0).TotalSeconds, 1)
Add-Content -Path $rolloutLog -Value "", "# end      : $($t1.ToString('s'))  exit=$rolloutCode  wall=${wall}s" -Encoding UTF8

if (-not (Test-Path $datasetDir)) {
    Write-Host "録画のフォルダが見つかりません: $datasetDir" -ForegroundColor Yellow
    $datasetDir = ""
}
if ($rolloutCode -ne 0) {
    Write-Host "lerobot-rollout が終了コード $rolloutCode で終わりました。ログ: $rolloutLog" -ForegroundColor Yellow
    if ((Read-Host "腕は動きましたか?結果を記録しますか? [y/N]").Trim().ToLower() -ne "y") {
        Write-Host "記録せずに終了します(ログは残っています)。"
        exit 1
    }
}

# ---- 4. 結果 ----
Write-Step "4. 結果"
Write-Host "  1) 〇   2) △   3) ×"
$result = $null
while (-not $result) {
    switch ((Read-Host "番号を入力").Trim()) {
        "1" { $result = "〇" }
        "2" { $result = "△" }
        "3" { $result = "×" }
        default { Write-Host "1 / 2 / 3 で入力してください。" -ForegroundColor Yellow }
    }
}

# ---- 5. ボタンを押すまでの時間 ----
Write-Step "5. ボタンを押すまでの時間"
$pressTime = $null
while ($null -eq $pressTime) {
    $ans = (Read-Host "秒で入力 (押せていなければ空のまま Enter)").Trim()
    if ($ans -eq "") { $pressTime = ""; break }
    $v = 0.0
    if ([double]::TryParse($ans, [ref]$v) -and $v -ge 0) { $pressTime = $v }
    else { Write-Host "数値で入力してください。" -ForegroundColor Yellow }
}

# ---- 6. 備考 ----
Write-Step "6. 備考"
$notes = (Read-Host "備考 (なければ Enter)").Trim()

# ---- 記録 ----
$row = [pscustomobject][ordered]@{
    trial_id        = $trialId
    datetime        = $t0.ToString("yyyy-MM-dd HH:mm:ss")
    model           = $model.Name
    policy_type     = $policyType
    model_trial_no  = $trialNo
    result          = $result
    press_time_s    = $pressTime
    notes           = $notes
    episode_time_s  = $EpisodeTime
    device          = $Device
    task            = $Task
    home_err        = $homeErr
    rollout_exit    = $rolloutCode
    wall_time_s     = $wall
    lerobot_commit  = $commit
    dataset_dir     = $datasetDir
    log_dir         = $trialDir
}
$lines = $row | ConvertTo-Csv -NoTypeInformation
if (-not (Test-Path $ResultsCsv)) { $lines[0] | Out-File -FilePath $ResultsCsv -Encoding utf8 }  # 先頭に BOM(Excel で文字化けしない)
Add-Content -Path $ResultsCsv -Value $lines[1] -Encoding UTF8
$row | ConvertTo-Json | Set-Content -Path (Join-Path $trialDir "trial.json") -Encoding UTF8

# スプレッドシートへ(失敗したら pending_sheet.jsonl に残し、次回送り直す)
Send-Pending
$sheetRow = [ordered]@{}
foreach ($p in $row.PSObject.Properties) { $sheetRow[$p.Name] = $p.Value }
if (Send-Sheet @{ row = $sheetRow }) { $sheetMsg = "スプレッドシートに書きました" }
else {
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $Pending -Encoding UTF8
    $sheetMsg = "スプレッドシートには未送信(次回の実行か -TestSheet で送り直します)"
}

# このモデルの集計
$rows = @(Import-Csv $ResultsCsv -Encoding UTF8 | Where-Object { $_.model -eq $model.Name })
$o  = @($rows | Where-Object { $_.result -eq "〇" }).Count
$tr = @($rows | Where-Object { $_.result -eq "△" }).Count
$x  = @($rows | Where-Object { $_.result -eq "×" }).Count
Write-Step "記録しました"
Write-Host "$($model.Name): $result / ${pressTime} 秒 / $notes"
Write-Host "累計 $($rows.Count) 回 (〇$o △$tr ×$x)"
Write-Host $sheetMsg
Write-Host "控え: $ResultsCsv"
Write-Host "ログ: $trialDir"
