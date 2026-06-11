"""
analyze_histories.py
--------------------
Read streamed train_lora history JSONs (std_r*_*.json + evo_*.json) and report
the evolutionary-vs-standard story from disk -- no model needed.

Each history dict (one run) holds aligned-to-`step` lists:
  step, train_acc, val_id, val_ood, dropout, wd        (one value per eval round)
  train_by_level: {level: [...]}                        (per-level train exact-match)
and sparse (step, value) pair-lists:
  loss, grad_norm, eff_rank

Usage:
  python analyze_histories.py [dir]      # dir defaults to ./histories then .
"""
import glob
import json
import os
import sys


def find_dir():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for d in ("histories", "."):
        if glob.glob(os.path.join(d, "std_r*_*.json")) or glob.glob(os.path.join(d, "evo_*.json")):
            return d
    return "."


def load(path):
    with open(path) as f:
        return json.load(f)


def pairs(seq):
    """[(step, v), ...] (possibly with None v) -> (steps, vals) dropping None vals."""
    xs, ys = [], []
    for p in seq or []:
        if p is None:
            continue
        s, v = p
        if v is not None:
            xs.append(s); ys.append(v)
    return xs, ys


def argmax(vals):
    best_i, best_v = -1, float("-inf")
    for i, v in enumerate(vals):
        if v is not None and v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


def pct(x):
    return "  -  " if x is None else f"{x:5.1%}"


def time_to(steps, vals, thresh):
    """First step where vals >= thresh, else None."""
    for s, v in zip(steps, vals):
        if v is not None and v >= thresh:
            return s
    return None


def summarize(name, h):
    steps   = h.get("step", [])
    vood    = h.get("val_ood", [])
    vid     = h.get("val_id", [])
    tacc    = h.get("train_acc", [])
    tbl     = h.get("train_by_level", {})
    lstep, lval = pairs(h.get("loss"))
    estep, eval_ = pairs(h.get("eff_rank"))

    i_vood, peak_vood = argmax(vood)
    i_vid,  peak_vid  = argmax(vid)
    final_step = steps[-1] if steps else 0

    print(f"\n{'='*78}\n{name}   ({len(steps)} eval rounds, final step {final_step})\n{'='*78}")
    print(f"  val_ood : peak {pct(peak_vood)} @ step {steps[i_vood] if i_vood>=0 else '-'}"
          f"   final {pct(vood[-1] if vood else None)}")
    print(f"  val_id  : peak {pct(peak_vid)}  @ step {steps[i_vid] if i_vid>=0 else '-'}"
          f"   final {pct(vid[-1] if vid else None)}")
    print(f"  train   : final {pct(tacc[-1] if tacc else None)}  (overall)")
    if tbl:
        finals = {l: (tbl[l][-1] if tbl[l] else None) for l in sorted(tbl, key=lambda x: int(x))}
        print("            per level (final):  " + "  ".join(f"L{l} {pct(v)}" for l, v in finals.items()))
    if lval:
        print(f"  loss    : first {lval[0]:.3f} -> final {lval[-1]:.3f}")
    if eval_:
        print(f"  eff_rank: first {eval_[0]:5.1f} -> min {min(eval_):5.1f} -> final {eval_[-1]:5.1f}"
              f"   (collapse {eval_[0]-min(eval_):+.1f})")
    # generalization gap and time-to-generalize
    if tacc and vood:
        print(f"  gap     : final train {pct(tacc[-1])} - val_ood {pct(vood[-1])} "
              f"= {(tacc[-1]-vood[-1]):+.1%}  (high = memorizing)")
    for thr in (0.10, 0.25, 0.50):
        t = time_to(steps, vood, thr)
        if t is not None:
            print(f"  reaches val_ood>={thr:.0%} at step {t}")
    return {"name": name, "peak_vood": peak_vood, "final_vood": vood[-1] if vood else None,
            "peak_vid": peak_vid, "eff_first": eval_[0] if eval_ else None,
            "eff_min": min(eval_) if eval_ else None}


def main():
    d = find_dir()
    std_files = sorted(glob.glob(os.path.join(d, "std_r*_*.json")),
                       key=lambda f: int(os.path.basename(f).split("_")[1][1:]))
    evo_files = sorted(glob.glob(os.path.join(d, "evo_*.json")))
    print(f"reading from: {os.path.abspath(d)}")
    print(f"std runs: {[os.path.basename(f) for f in std_files]}")
    print(f"evo runs: {[os.path.basename(f) for f in evo_files]}")
    if not std_files and not evo_files:
        print("\n!! no history files found. Drop std_r*_*.json / evo_*.json here and rerun.")
        return

    rows = []
    for f in std_files:
        r = os.path.basename(f).split("_")[1]            # 'r256'
        rows.append(summarize(f"standard {r}", load(f)))
    evo_row = None
    for f in evo_files:
        evo_row = summarize(f"EVOLUTIONARY  ({os.path.basename(f)})", load(f))

    # ---- verdict -----------------------------------------------------------
    print(f"\n{'#'*78}\n# VERDICT\n{'#'*78}")
    if rows:
        best = max(rows, key=lambda r: (r["peak_vood"] if r["peak_vood"] is not None else -1))
        print(f"  best standard on val_ood : {best['name']}  peak {pct(best['peak_vood'])}")
    if evo_row:
        print(f"  evolutionary on val_ood  : peak {pct(evo_row['peak_vood'])}")
        if rows and evo_row["peak_vood"] is not None and best["peak_vood"] is not None:
            delta = evo_row["peak_vood"] - best["peak_vood"]
            print(f"  evo - best_std (peak OOD): {delta:+.1%}  "
                  f"-> {'EVO WINS' if delta > 0 else 'no evo advantage'}")
        if evo_row["eff_first"] is not None and evo_row["eff_min"] is not None:
            print(f"  evo eff-rank collapse    : {evo_row['eff_first']:.1f} -> {evo_row['eff_min']:.1f} "
                  f"({evo_row['eff_first']-evo_row['eff_min']:+.1f})")


if __name__ == "__main__":
    main()
