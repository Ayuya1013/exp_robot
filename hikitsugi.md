# 引継ぎ(2026-11-02 時点):SO-101 で「エアコンの運転/停止ボタンを押す」

課題:SO-101 アームで、壁のエアコンのリモコンの「運転/停止」を押す(タスク文 `Press the power button`)。
LeRobot 0.6.2 で ACT と smolVLA を学習し、**実データとシミュ(MuJoCo)のデータの混ぜ方**で実機の成功率がどう変わるかを比べている。

---

## 1. 現在の状況

### 環境

| 場所 | 内容 |
|---|---|
| 実機の PC(Windows) | `C:\Users\yuya\lerobot`(LeRobot 本体、venv は `.venv`)。follower = COM4、leader = COM5。カメラ wrist = index 0、top = index 1(どちらも 640x480、MJPG、30fps)。GPU なし(推論は CPU) |
| シミュ一式(Windows) | `C:\Users\yuya\lerobot\sim`(自分の GitHub の private リポジトリ so101-sim)。SO-101 のモデルは `C:\Users\yuya\SO-ARM100\Simulation\SO101` |
| GPU サーバー | `akamatsu@montecarlo`、作業場所 `~/exp/robot`、venv は `~/exp/robot/lerobot/.venv` |
| サーバーの GPU | 0 = Quadro P400(2GB、**学習に使えない**)/ 1 = GP100(16GB)/ 2 = GV100(32GB、一番速い)。他のユーザーと共有。番号は `CUDA_DEVICE_ORDER=PCI_BUS_ID` で固定 |

サーバーの詳しいファイル構成は `~/exp/robot/SUMMARY_20261102.md`、smolVLA の一括学習の使い方は `NOTES_20261030.md` / `NOTES_20261031.md`。

### データセット(サーバーの `~/exp/robot/datasets/`)

| 名前 | 中身 |
|---|---|
| `ac_power_button_v3` | 実データ 50 本、14,372 フレーム(**これを実データとして使う**。`ac_power_button` は最初の 20 本版なので使わない) |
| `ac_power_sim_t1` | シミュ 200 本、44,976 フレーム(ランダム化の強さ θ=1、乱数の種 1)。各エピソードの設定は `ac_power_sim_t1_sim_params.jsonl` |
| `ac_power_mix_x1 / x3 / x9` | v3 を 1 / 3 / 9 回重ねて、シミュ 200 本と結合(実データが先頭) |

### ACT:11 条件すべて学習済み(30000 ステップ、バッチ 8)

| 系列 | 条件(出力は `lerobot/outputs/train/<名前>_s30000`) | 中身 |
|---|---|---|
| sim | `act_sim50 / 100 / 150 / 200` | シミュだけ(先頭 N 本) |
| mix-B | `act_mixB_sim00 / 25 / 50 / 75` | 実 50 本 + シミュ。フレームに占めるシミュの割合 0 / 25.3 / 50.0 / 75.8% |
| mix-C | `act_mixC_real25 / 50 / 75` | シミュ 200 本 + 実を ×1 / ×3 / ×9。実の割合 24.2 / 48.9 / 74.2% |

注意:`act_mixC_real25` と `act_mixB_sim75` はデータも乱数の種も同じ(独立な 2 回目ではない)。

### smolVLA:学習中(`train_smolvla.py`、smolvla_base から更新 20000 回、実効バッチ 64)

11/2 12:43 時点。済:`mixB_sim00 / sim25 / sim50`。学習中:`mixB_sim75`(GPU 2、11/2 23:30 ごろ終了)、`mixC_real50`(GPU 1、11/4 10:00 ごろ)。
未:`mixC_real75`、`sim50 / 100 / 150 / 200`。**全部終わるのは 11/8 朝ごろの見込み**。
各 GPU の `run` が、終わると次の条件を自動で取る。

```bash
cd ~/exp/robot && source lerobot/.venv/bin/activate
python train_smolvla.py plan                                   # 状態
echo ckpt > logs/smolvla_stop_g2                               # 次のチェックポイントで止める(10/31 に修正、未検証)
nohup python train_smolvla.py run --gpu 2 >> logs/train_smolvla_g2.log 2>&1 &   # 再開
```

比較のときの注意:`smolvla_mixB_sim50` は GP100 で 16×4(更新回数と実効バッチは同じ)。`mixB_sim00 / sim25` は止め方のバグで一部を学習し直した。

### 実機評価:始めたところ

- 今やっていること:**act_mixC_real75(期待値が一番高い)と act_mixB_sim00(実データだけ)を各 5 回**、順番 A B B A A B B A A B。
  1 回目は名前のエラーで止まったので、record.ps1 を直してやり直し中
- モデルの置き場所:`C:\Users\yuya\lerobot\models\<名前>`(`config.json` と `model.safetensors` が直接入っているフォルダ)。今あるのは上の 2 本だけ
- 実行(`C:\Users\yuya\lerobot\sim\scripts` で):`.\record.ps1 -Policy $A -Episodes 1`
- record.ps1 は直してある:`$WristRotation = 0`(180 だと手先の画像が逆さま)、`$EpisodeTime = 20`、
  評価データの名前を `rollout_...` に(lerobot-rollout の決まり)。初回は `Unblock-File .\record.ps1` が必要だった
- 毎回の試行の前に:腕を手で畳んだ待機姿勢に戻す、エアコンを同じ状態(停止)にする
- 記録:PowerShell の関数 `rec`(model, 成功 1/0, 押すまでの秒, 失敗の種類, メモ)で
  `C:\Users\yuya\lerobot\sim\eval\real_eval_act.csv` に 1 行ずつ追記。失敗の種類は stuck / reach / offset / wrong / hit。
  評価の動画は `~/.cache/huggingface/lerobot/local/rollout_*` に残る

### シミュ評価:道具はできている(未実行)

- `sim\sim_eval.py`:学習済み ACT を MuJoCo の中で動かす。全モデルで同じ場面(学習に使っていない乱数の種)。
  θ=1・2 は各 20 回、θ=0 は 1 回(毎回同じになるため)。成功 / 別のボタン / 当たった / 時間切れ、押すまでの時間、一番近づいた距離を CSV に記録。
  お手本(IK)で動かす `--stub` では確認済み。**ACT を読み込む部分(`ACTRunner`)は未確認**
- `sim\real_eval_sheet.py`:11 本 × 5 回の実機評価の順番表(ラウンドごとにランダム)

---

## 2. これからやること(優先順)

1. **ACT の 2 本の実機評価を終える**(各 5 回)。`real_eval_act.csv` から成功率・押すまでの時間・失敗の種類をまとめる
2. **ACT 11 本のシミュ評価**(Windows、夜に回せる)
   - 残り 9 本をサーバーから `models\` に持ってくる(各 `checkpoints/last/pretrained_model` を `models\<名前>` に。`_s30000` は付けない)
   - 確認:`python sim_eval.py --stub --thetas 1 --episodes 2 --out sim_eval_test`
   - 1 本で試す:`python sim_eval.py --models ..\models --only act_mixB_sim00 --thetas 1 --episodes 2 --video_n 1 --out sim_eval_test`
     (エラーが出たら `ACTRunner` を LeRobot 0.6.2 の `lerobot/inference/sync.py` に合わせて直す)
   - 本番:`python sim_eval.py --models ..\models --thetas 0 1 2 --episodes 20 --video_n 2 --out sim_eval_act`(途中から再開できる)
3. **ACT 11 本の実機評価**(各 5 回、`python real_eval_sheet.py --models ..\models --rounds 5` の順番で)。
   10/16 の `offline_eval.py`(動き出しの評価)も 11 本に当てて、「動き出さない」問題が改善したかを見る
4. **シミュと実機の成績を比べる**:モデルごとの成功率の順位がどれだけ揃うか(シミュ評価は実データだけのモデルに不利なので、実機の代わりではなく、揃い方を調べる材料)
5. **ACT の結果で smolVLA の残り 6 本の順番を見直す**(例:sim だけの条件が実機で全く押せなければ後回し)。未着手のうちに `train_smolvla.py` の条件の順番を変える
6. **`echo ckpt` の修正を試す**:GPU 2 がチェックポイントを保存した直後に出し、次のチェックポイントまで止まらなければ成功
7. **smolVLA の評価**:終わった条件から。CPU では遅いので、GV100 の空きメモリで推論サーバーを動かす
8. **次のシミュのデータ生成の改善**(必要になったら):
   - 押し込みの目標を浅く(3mm → 1mm)。今のデータは約 1/4 で、押したまま指先がパネル面を上に約 12mm 滑っている
   - θ=0・2 のデータを作り、ランダム化の強さの比較をする
   - 生成の並列化(乱数の種を変えて複数プロセス → 結合)。1 エピソード約 44 秒で、遅いのは CPU 側
   - 完了の判定用に、シミュに運転ランプの点灯・消灯とバックライトの時間を足す(俯瞰カメラは方策の入力ではなく、完了の判定に使う予定)
9. **記録**:結果を研究記録(Google ドキュメント「赤松さんプロジェクト」、ID `1qFiz4eiYmm6tvsy4b25anHyS24cIlDT1HoaBltmg7qI`)に書く。
   メインページの表(タイトル / コメント / 短題)に 1 行と、本文に「YYYY/MM/DD 研究記録」の見出し。環境の事実は「環境リファレンス」タブへ

---

## 3. はまりやすい所

- サーバーで GPU 0 を指定しない(2GB でメモリ不足になる)。学習の前に `nvidia-smi` で空きを確かめる
- 実行中のシェルスクリプトを上書きしない(動いている学習が構文エラーで止まる)。新しい版は別の名前で置く
- データセットを結合する前に `meta/info.json` の `total_episodes` を確かめる
- 実機の評価は、毎回同じ待機姿勢・同じエアコンの状態から始める(モデルどうしの比較が崩れる)