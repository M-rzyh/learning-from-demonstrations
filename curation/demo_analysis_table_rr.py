"""ACF, USR and demo return for the random/random (rr) arms: both human and
synthetic demonstrations drawn per seed, exactly as the rr runs draw them
(ds.shuffle(seed) then first N). Metric code imported from
demo_analysis_c1_vs_e4.py; the ranked human arm is included for reference.

Writes a CSV; nothing existing is modified.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from datasets import load_from_disk
import datasets; datasets.disable_progress_bar()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_analysis_c1_vs_e4 import (action_change_frequency,
                                    unique_discretized_state_ratio, DEMOS, EXPERT, N_BINS)

def stats(ds, bins):
    acf = [action_change_frequency(t["acts"]) for t in ds]
    usr = [unique_discretized_state_ratio(t["obs"], bins)[0] for t in ds]
    ret = [float(np.sum(t["rews"])) for t in ds]
    return float(np.nanmean(acf)), float(np.nanmean(usr)), float(np.mean(ret))

def ranked(n):
    if n <= 10:
        return load_from_disk(str(DEMOS / "session_3_ext700_ret_only_top10")).select(range(n))
    return load_from_disk(str(DEMOS / f"session_3_ext700_ret_only_top{n}"))

p = argparse.ArgumentParser()
p.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 10, 50, 100, 250, 400])
p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

hp = load_from_disk(str(DEMOS / "session_3_ext700_ret_gt200"))
sp = load_from_disk(str(EXPERT))
print(f"human pool {len(hp)}, expert pool {len(sp)}")
sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
rows = []
for n in a.budgets:
    H = np.array([stats(hp.shuffle(seed=s).select(range(min(n, len(hp)))), N_BINS) for s in a.seeds])
    S = np.array([stats(sp.shuffle(seed=s).select(range(min(n, len(sp)))), N_BINS) for s in a.seeds])
    K = stats(ranked(n), N_BINS)
    rows.append(dict(N=n,
        h_acf=H[:,0].mean(), h_acf_sd=sd(H[:,0]), h_usr=H[:,1].mean(), h_usr_sd=sd(H[:,1]),
        h_ret=H[:,2].mean(), h_ret_sd=sd(H[:,2]),
        s_acf=S[:,0].mean(), s_acf_sd=sd(S[:,0]), s_usr=S[:,1].mean(), s_usr_sd=sd(S[:,1]),
        s_ret=S[:,2].mean(), s_ret_sd=sd(S[:,2]),
        k_acf=K[0], k_usr=K[1], k_ret=K[2],
        acf_ratio_s_over_h=S[:,0].mean()/H[:,0].mean(),
        acf_ratio_s_over_k=S[:,0].mean()/K[0],
        usr_ratio_h_over_s=H[:,1].mean()/S[:,1].mean(),
        usr_ratio_k_over_s=K[1]/S[:,1].mean()))
    r = rows[-1]
    print(f"N={n:<4} ACF h={r['h_acf']:.4f}+-{r['h_acf_sd']:.4f} s={r['s_acf']:.4f}+-{r['s_acf_sd']:.4f} "
          f"k={r['k_acf']:.4f} | USR h={r['h_usr']:.4f}+-{r['h_usr_sd']:.4f} s={r['s_usr']:.4f}+-{r['s_usr_sd']:.4f} "
          f"k={r['k_usr']:.4f} | ret h={r['h_ret']:.1f}+-{r['h_ret_sd']:.1f} s={r['s_ret']:.1f}+-{r['s_ret_sd']:.1f}")
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
agg = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != "N"}
print("\navg over N: ACF h=%.4f(sd across N %.4f) s=%.4f(%.4f) k=%.4f | ratios s/h=%.2f s/k=%.2f" % (
    agg["h_acf"], np.std([r["h_acf"] for r in rows], ddof=1), agg["s_acf"],
    np.std([r["s_acf"] for r in rows], ddof=1), agg["k_acf"], agg["acf_ratio_s_over_h"], agg["acf_ratio_s_over_k"]))
print("            USR h=%.4f(%.4f) s=%.4f(%.4f) k=%.4f | ratios h/s=%.2f k/s=%.2f" % (
    agg["h_usr"], np.std([r["h_usr"] for r in rows], ddof=1), agg["s_usr"],
    np.std([r["s_usr"] for r in rows], ddof=1), agg["k_usr"], agg["usr_ratio_h_over_s"], agg["usr_ratio_k_over_s"]))
print(f"wrote {a.out}")
