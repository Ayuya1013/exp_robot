# 実機評価(SO-101 / Press the power button)

## 使い方(PowerShell で、このフォルダにて)

```powershell
.\run_trial.ps1             # 1 回評価する
.\run_trial.ps1 -SaveHome   # ホーム姿勢を保存し直す(初回は自動で聞かれる)
```

流れ:モデル選択 → ホーム姿勢へ移動 → 1 回実行(record.ps1 の -Policy と同じ条件、20 秒、CPU)
→ 結果(〇 / △ / ×)→ ボタンを押すまでの秒 → 備考 → Google スプレッドシートと `results.csv`(控え)に 1 行追記。

## スプレッドシートの準備(最初に 1 回)

1. 記録用のスプレッドシートを作り、「拡張機能 → Apps Script」を開く
2. `sheets_webapp.gs` の中身を全部貼り付けて保存
3. 「デプロイ → 新しいデプロイ」→ 種類「ウェブアプリ」、実行ユーザー「自分」、アクセスできるユーザー「全員」→ デプロイ(初回は権限の許可を求められる)
4. 表示されたウェブアプリの URL(`https://script.google.com/macros/s/.../exec`)を `sheets_config.json` の `url` に貼る
5. `.\run_trial.ps1 -TestSheet` で「接続できました」と出れば完了

- 書き込み先はシート `results`(なければ自動で作る)。1 行目の見出しの名前で列を合わせるので、列の並べ替えや自分用の列の追加をしてもよい
- 送れなかった行は `pending_sheet.jsonl` に残り、次の実行時(または `-TestSheet`)に送り直す。同じ `trial_id` は二重に書かない
- `.gs` のコードを直したら「デプロイを管理 → 編集 → バージョン: 新バージョン」で更新する(URL は変わらない)
- `sheets_config.json` の token は秘密にする(URL と token があれば誰でも書き込める)

## ファイル

| ファイル                | 中身                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `results.csv`           | 全試行の結果(1 行 = 1 試行、Excel / Google スプレッドシートでそのまま開ける)                               |
| `logs/<日時>_<モデル>/` | `home.log`(ホーム移動)、`rollout.log`(lerobot-rollout の全出力と実行コマンド)、`trial.json`(記録した 1 行) |
| `home_pose.json`        | ホーム姿勢の関節角                                                                                         |
| `go_home.py`            | ホーム姿勢への移動と保存                                                                                   |

評価の動画は `~/.cache/huggingface/lerobot/local/rollout/<モデル名>/rollout_eval_<試行ID>/` に残り(`HF_LEROBOT_HOME` を設定していればその下の `local/rollout/`)、パスは `results.csv` の `dataset_dir` 列に書かれる。

## 列(スプレッドシート・results.csv 共通)

`trial_id, datetime, model, policy_type, model_trial_no, result(〇△×), press_time_s, notes,
episode_time_s, device, task, home_err(ホーム到達誤差), rollout_exit, wall_time_s, lerobot_commit, dataset_dir, log_dir`
