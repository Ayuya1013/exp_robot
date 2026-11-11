"""実機評価の順番表(記録用のシート)を作る。各回(ラウンド)で全モデルを1回ずつ、順番はラウンドごとにランダム。

  python real_eval_sheet.py --models C:\\Users\\yuya\\lerobot\\models --rounds 5
  → real_eval_sheet.csv(Excel で開いて、成功・押すまでの時間・失敗の種類を書き込む)

失敗の種類の書き方: reach(届かない)/ offset(ずれて押せない)/ wrong(別のボタン)/ hit(壁やパネルに当たる)/ stuck(動かない・止まる)
"""

import argparse
import csv
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_eval_sheet.csv")
    args = ap.parse_args()
    models = sorted(
        d for d in os.listdir(args.models) if os.path.exists(os.path.join(args.models, d, "config.json"))
    )
    rng = random.Random(args.seed)
    rows, n = [], 0
    for r in range(1, args.rounds + 1):
        order = models[:]
        rng.shuffle(order)
        for m in order:
            n += 1
            rows.append(
                dict(
                    no=n,
                    round=r,
                    model=m,
                    path=os.path.join(args.models, m),
                    success="",
                    t_press_s="",
                    failure="",
                    note="",
                )
            )
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(models)} モデル × {args.rounds} 回 = {len(rows)} 試行: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
