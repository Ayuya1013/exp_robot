# save.ps1 - sim フォルダの変更をまとめて GitHub に上げる
#   .\save.ps1                    : 日時をメッセージにして保存
#   .\save.ps1 "やったことのメモ"  : メッセージを指定
param([string]$Msg = "")
Set-Location $PSScriptRoot
if ($Msg -eq "") { $Msg = "作業の保存 " + (Get-Date -Format "yyyy-MM-dd HH:mm") }
git add -A
git status --short
git commit -m $Msg
git push