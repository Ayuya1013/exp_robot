"""2つの _sim_params.jsonl を1本ずつ比べ、パネルの位置・向き・動き・押し込み量が一致するかを確かめる。

使い方(PowerShell):
  python compare_params.py <t1 の jsonl> <vis0 の jsonl>
例:
  python compare_params.py $env:USERPROFILE\\.cache\\huggingface\\lerobot\\local\\dataset\\sim\\ac_power_sim_t1_sim_params.jsonl `
                           $env:USERPROFILE\\.cache\\huggingface\\lerobot\\local\\dataset\\sim\\ac_power_sim_vis0_sim_params.jsonl
"""
import json
import sys

KEYS = ["seed", "theta", "panel_dpos_mm", "panel_drpy_deg", "motion", "depth_mm"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    n = min(len(a), len(b))
    print(f"エピソード数: 1つ目 {len(a)}、2つ目 {len(b)}。先頭 {n} 本を比べます。")
    bad = 0
    for i in range(n):
        diff = [k for k in KEYS if a[i].get(k) != b[i].get(k)]
        if diff:
            bad += 1
            if bad <= 5:
                print(f"  エピソード {i}: 違う項目 {diff}")
                for k in diff:
                    print(f"    {k}: {a[i].get(k)}  /  {b[i].get(k)}")
    fa = sum(r["motion"]["frames"] for r in a[:n])
    fb = sum(r["motion"]["frames"] for r in b[:n])
    print(f"フレーム数の合計: {fa} / {fb}")
    if bad == 0:
        print(f"OK: {n} 本すべて、パネルの位置・向き・動き・押し込み量が一致しました。")
    else:
        print(f"一致しないエピソードが {bad} 本あります(最初の5本だけ表示)。")
        print("1本目からずれるなら、--seed・--theta か、MOTION・RANDOM・random.json が t1 の生成時と違う可能性があります。")


if __name__ == "__main__":
    main()