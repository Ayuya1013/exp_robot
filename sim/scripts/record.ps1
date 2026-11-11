# record.ps1 - SO-101 demo recording helper
#   .\record.ps1            : 本番(20回)。同名フォルダがあれば選択肢を出す
#   .\record.ps1 -Test      : 試し録画(2回)。毎回フォルダを消して最初から
#   .\record.ps1 -Resume    : 既存データに続けて録画(-Episodes は追加する回数)
#   .\record.ps1 -Fresh     : 既存データを確認なしで削除して最初から
#   .\record.ps1 -Episodes 5 : 回数を指定
#   .\record.ps1 -Policy <モデルのフォルダ> : 学習したモデルを実機で動かして評価(lerobot-rollout を使用。毎回、新しいフォルダに保存)
#   .\record.ps1 -Show     : カメラ映像(Rerun)を表示。-Test のときは自動で表示

param(
    [switch]$Test,
    [switch]$Resume,
    [switch]$Fresh,
    [switch]$Show,
    [string]$Policy = "",
    [int]$Episodes = 0
)

# ===== 設定(ここだけ直す) =====
$FollowerPort  = "COM4"
$LeaderPort    = "COM5"
$Name          = "ac_power_button_v3"
$Task          = "Press the power button"
$EpisodeTime   = 20
$ResetTime     = 15
$WristRotation = 0      # 手先カメラの回転。回さないなら 0
$LerobotDir    = "C:\Users\yuya\lerobot"
# ==============================

# 保存先: 録画(お手本)は local\dataset\real\<Name>\、モデルの評価(-Policy)は local\rollout\<モデル名>\rollout_eval_<日時>_<モデル名>\
$LerobotHome = if ($env:HF_LEROBOT_HOME) { $env:HF_LEROBOT_HOME } else { Join-Path $env:USERPROFILE ".cache\huggingface\lerobot" }
$RealDir     = Join-Path $LerobotHome "local\dataset\real"
$RolloutDir  = Join-Path $LerobotHome "local\rollout"

$exe =Join-Path $LerobotDir ".venv\Scripts\lerobot-record.exe"
if (-not (Test-Path $exe)) {
    Write-Host "lerobot-record.exe が見つかりません: $exe" -ForegroundColor Red
    exit 1
}

if ($Policy -ne "") {
    if (-not (Test-Path $Policy)) {
        Write-Host "モデルが見つかりません: $Policy" -ForegroundColor Red
        exit 1
    }
    # モデル名: models\<名前>\ ならその名前。学習の出力 ...\<run>\checkpoints\<step>\pretrained_model なら <run>
    $policyItem = Get-Item $Policy
    $ModelName = $policyItem.Name
    if ($ModelName -eq "pretrained_model") { $ModelName = $policyItem.Parent.Parent.Parent.Name }
    $Name = "rollout_eval_" + (Get-Date -Format "yyyyMMdd_HHmmss") + "_$ModelName"   # run_trial.ps1 と同じ命名
    $Show = $true
    if ($Episodes -le 0) { $Episodes = 10 }
}

if ($Test) {
    $Name = "ac_power_test"
    $Fresh = $true
    $Show = $true
    if ($Episodes -le 0) { $Episodes = 2 }
}
elseif ($Episodes -le 0) {
    $Episodes = 20
}

# COMポートが見えているか確認
$ports = [System.IO.Ports.SerialPort]::GetPortNames()
foreach ($p in @($FollowerPort, $LeaderPort)) {
    if ($ports -notcontains $p) {
        Write-Host "${p} が見つかりません。現在のポート: $($ports -join ', ')" -ForegroundColor Red
        Write-Host "lerobot-find-port で確認して、スクリプト冒頭のポート番号を直してください。"
        exit 1
    }
}

# 既存データセットの扱い
$Root = if ($Policy -ne "") { Join-Path (Join-Path $RolloutDir $ModelName) $Name } else { Join-Path $RealDir $Name }
New-Item -ItemType Directory -Force (Split-Path $Root) | Out-Null   # 親フォルダだけ作る(データセットのフォルダは lerobot が作る)
$ResumeFlag = $false

if (Test-Path $Root) {
    if ($Resume) {
        $ResumeFlag = $true
    }
    elseif ($Fresh) {
        Write-Host "既存フォルダを削除します: $Root"
        Remove-Item -Recurse -Force $Root
    }
    else {
        Write-Host "既にデータセットがあります: $Root" -ForegroundColor Yellow
        $ans = Read-Host "[D]削除して最初から / [R]続きから追加 / [N]別名(日時付き)で新規 / [Q]中止"
        switch ($ans.ToUpper()) {
            "D" { Remove-Item -Recurse -Force $Root }
            "R" { $ResumeFlag = $true }
            "N" {
                $Name = "${Name}_" + (Get-Date -Format "yyyyMMdd_HHmmss")
                $Root = Join-Path (Split-Path $Root) $Name
            }
            default { Write-Host "中止しました。"; exit 0 }
        }
    }
}
elseif ($Resume) {
    Write-Host "続き元のデータセットがありません: $Root" -ForegroundColor Red
    exit 1
}

$displayValue = if ($Show) { "true" } else { "false" }

# カメラ設定
$wrist = "wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG"
if ($WristRotation -ne 0) { $wrist += ", rotation: $WristRotation" }
$wrist += "}"
$top  = "top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}"
$cams = "{" + $wrist + ", " + $top + "}"

$recordArgs = @(
    "--robot.type=so101_follower",
    "--robot.port=$FollowerPort",
    "--robot.id=my_follower",
    "--robot.cameras=$cams",
    "--teleop.type=so101_leader",
    "--teleop.port=$LeaderPort",
    "--teleop.id=my_leader",
    "--display_data=$displayValue",
    "--dataset.repo_id=local/$Name",
    "--dataset.root=$Root",
    "--dataset.no_stamp=true",
    "--dataset.single_task=$Task",
    "--dataset.num_episodes=$Episodes",
    "--dataset.episode_time_s=$EpisodeTime",
    "--dataset.reset_time_s=$ResetTime",
    "--dataset.push_to_hub=false"
)
if ($ResumeFlag) { $recordArgs += "--resume=true" }
if ($Policy -ne "") {
    # 学習済みモデルの実行は lerobot-record ではなく lerobot-rollout を使う
    $exe = Join-Path $LerobotDir ".venv\Scripts\lerobot-rollout.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "lerobot-rollout.exe が見つかりません: $exe" -ForegroundColor Red
        exit 1
    }
    # --dataset.root と --dataset.no_stamp=true はそのまま渡す(名前に日時は付けてあるので、lerobot に付け足させない)
    $recordArgs += "--strategy.type=episodic"
    $recordArgs += "--policy.path=$Policy"
    $recordArgs += "--device=cpu"
}

Write-Host "保存先: $Root"
Write-Host "録画回数: $Episodes  (続きから: $ResumeFlag / 映像表示: $displayValue)"
& $exe @recordArgs

# 結果の確認
$info = Join-Path $Root "meta\info.json"
if (Test-Path $info) {
    try {
        $j = Get-Content $info -Raw | ConvertFrom-Json
        Write-Host "保存済みエピソード数: $($j.total_episodes) / フレーム数: $($j.total_frames)" -ForegroundColor Green
    }
    catch {
        Write-Host "info.json は読めませんでした: $info"
    }
}
else {
    Write-Host "info.json が見つかりません(保存されていない可能性があります): $info" -ForegroundColor Yellow
}