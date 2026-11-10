# collect_logs.ps1 - サーバーとクライアントのログを集めて1つのzipにまとめる
#   .\collect_logs.ps1                       : デスクトップに robot_logs_日時.zip を作る
#   .\collect_logs.ps1 -DriveDir "G:\マイドライブ\robot_logs" : 作ったzipをドライブのフォルダにもコピー
param([string]$Tag = (Get-Date -Format "yyyyMMdd_HHmmss"), [string]$DriveDir = "")

$root = "C:\Users\yuya\lerobot\logs"
$srv  = Join-Path $root "server"
New-Item -ItemType Directory -Force $srv | Out-Null

# サーバーのログを取得(パスワードを聞かれる)
scp "akamatsu@montecarlo:~/exp/robot/logs/*.log" $srv

$zip = Join-Path "$env:USERPROFILE\Desktop" "robot_logs_$Tag.zip"
Compress-Archive -Path "$root\*" -DestinationPath $zip -Force
Write-Host "作成: $zip"

if ($DriveDir -ne "") {
    New-Item -ItemType Directory -Force $DriveDir | Out-Null
    Copy-Item $zip $DriveDir
    Write-Host "コピー: $DriveDir"
}
