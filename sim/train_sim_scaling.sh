#!/usr/bin/env bash
# train_sim_scaling.sh - シミュのエピソード数を変えて ACT を順に学習する(GPUサーバー用)
#
#   MODE=sim : シミュのデータだけ。ac_power_sim_t1 の先頭 N エピソードを使う
#   MODE=mix : 実データ50 + シミュ N。結合した ac_power_mix_t1 を使う(実データは 0〜49、シミュは 50〜249)
#              N=0 を入れると、同じ学習量の「実データだけ」の比較になる
#
#   使い方(サーバーで):
#     nohup env MODE=sim GPU=2 bash train_sim_scaling.sh > ~/exp/robot/logs/train_scaling_sim.log 2>&1 &
#     nohup env MODE=mix GPU=2 NS="0 50 100 150 200" bash train_sim_scaling.sh > ~/exp/robot/logs/train_scaling_mix.log 2>&1 &
#
#   変えられるもの: STEPS(既定 30000)、BATCH(既定 8)、NS(既定 "50 100 150 200")、GPU(既定 2)
set -eu
MODE=${MODE:-sim}
GPU=${GPU:-2}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-8}
NS=${NS:-"50 100 150 200"}
D=$HOME/exp/robot/datasets
LOG=$HOME/exp/robot/logs

cd $HOME/exp/robot/lerobot
source .venv/bin/activate

if [ "$MODE" = "sim" ]; then
  REPO=local/ac_power_sim_t1; ROOT=$D/ac_power_sim_t1; REAL=0
elif [ "$MODE" = "mix" ]; then
  REPO=local/ac_power_mix_t1; ROOT=$D/ac_power_mix_t1; REAL=50
else
  echo "MODE は sim か mix"; exit 1
fi

for N in $NS; do
  # 使うエピソードの番号: 実データ 0〜REAL-1 と、シミュの先頭 N 本(50 ⊂ 100 ⊂ 150 ⊂ 200 と入れ子になる)
  EPS=$(python -c "import json; print(json.dumps(list(range($REAL)) + list(range($REAL, $REAL + $N)), separators=(',', ':')))")
  NAME=act_${MODE}${N}_s${STEPS}
  if [ -d outputs/train/$NAME ]; then
    echo "$(date '+%F %T') $NAME は既にあるので飛ばす"; continue
  fi
  echo "$(date '+%F %T') 開始: $NAME(実データ $REAL + シミュ $N エピソード、$STEPS ステップ、バッチ $BATCH)"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU lerobot-train \
    --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.episodes="$EPS" \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/$NAME --job_name=$NAME \
    --steps=$STEPS --batch_size=$BATCH --log_freq=100 --save_freq=5000 --wandb.enable=false \
    > $LOG/train_$NAME.log 2>&1
  echo "$(date '+%F %T') 終了: $NAME(ログ $LOG/train_$NAME.log)"
done