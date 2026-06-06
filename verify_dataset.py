"""
verify_dataset.py
-----------------
Spot-checks a sampled level of the pooled dataset by re-running each row's
operations in chain order and comparing computed answers to stored answers.

Usage:
  python verify_dataset.py                      # level 5, 1000 samples
  python verify_dataset.py --level 3 --samples 500
"""

import argparse
import csv
import math
import random


def mod_inverse(a, p):
    return pow(a, p - 2, p)


def apply_op(op, x, c):
    """Apply one encoded operation: op(x, c) → result."""
    if op.startswith("mod_+_"):
        p = int(op.rsplit("_", 1)[-1])
        return (x + c) % p
    elif op.startswith("mod_-_"):
        p = int(op.rsplit("_", 1)[-1])
        return (x - c) % p
    elif op.startswith("mod_x_"):
        p = int(op.rsplit("_", 1)[-1])
        return (x * c) % p
    elif op.startswith("mod_div_"):
        p = int(op.rsplit("_", 1)[-1])
        return (x * mod_inverse(c, p)) % p
    elif op == "2D+":
        return x + c
    elif op == "2D-":
        return x - c
    elif op == "2Dx":
        return x * c
    else:
        raise ValueError(f"Unknown op: {op!r}")


def format_chain(row, level):
    """Return a human-readable step-by-step description of one row's op chain."""
    if level == 1:
        ops = [row["task"]]
    else:
        ops = [row[f"op{i}"] for i in range(1, level + 1)]

    input_keys = ["a", "b", "c", "d", "e", "f"][: level + 1]
    inputs = [int(row[k]) for k in input_keys]

    lines = []
    lines.append(f"Operations ({level}): {' → '.join(ops)}")
    lines.append(f"Inputs ({level + 1}):     {', '.join(str(v) for v in inputs)}")
    lines.append("")

    y = inputs[0]
    for step, op in enumerate(ops, start=1):
        c = inputs[step]
        result = apply_op(op, y, c)
        lines.append(f"  Step {step}: {y:>6}  [{op}]  {c}  =  {result}")
        y = result

    lines.append("")
    lines.append(f"The final answer is: {int(row['answer'])}")
    return "\n".join(lines)


def verify_row(row, level):
    """
    Re-run the op chain and return (computed, expected, match).

    Chain for level N:
      y  = op1(a, b)
      y  = op2(y,  c)
      y  = op3(y,  d)
      ...
      y  = opN(y,  <last input>)
    """
    # Level 1 stores the op in 'task'; levels 2+ use op1..opN
    if level == 1:
        ops = [row["task"]]
    else:
        ops = [row[f"op{i}"] for i in range(1, level + 1)]

    # Inputs: a, b, c, d, e, f  (level+1 values)
    input_keys = ["a", "b", "c", "d", "e", "f"][: level + 1]
    inputs = [int(row[k]) for k in input_keys]

    y = apply_op(ops[0], inputs[0], inputs[1])
    for idx, op in enumerate(ops[1:], start=2):
        y = apply_op(op, y, inputs[idx])

    expected = int(row["answer"])
    return y, expected, y == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",     type=int, default=5)
    parser.add_argument("--samples",   type=int, default=1000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--input_dir", type=str, default="datasets")
    parser.add_argument("--show",      type=int, default=0,
                        help="Print this many rows as human-readable chains")
    args = parser.parse_args()

    path = f"{args.input_dir}/level_{args.level}.csv"
    print(f"\nLoading {path} ...")

    with open(path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rng = random.Random(args.seed)
    sample = rng.sample(all_rows, min(args.samples, len(all_rows)))

    print(f"Verifying {len(sample):,} rows (dataset size: {len(all_rows):,})\n")

    if args.show > 0:
        print(f"{'─' * 60}")
        for row in sample[: args.show]:
            print("Raw row:")
            print("  " + ", ".join(f"{k}={v}" for k, v in row.items() if v != ""))
            print()
            print(format_chain(row, args.level))
            print(f"{'─' * 60}")
        print()

    n_ok = n_fail = 0
    for row in sample:
        computed, expected, ok = verify_row(row, args.level)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if args.level == 1:
                ops = [row["task"]]
            else:
                ops = [row[f"op{i}"] for i in range(1, args.level + 1)]
            input_keys = ["a", "b", "c", "d", "e", "f"][: args.level + 1]
            inputs = {k: row[k] for k in input_keys}
            print(f"FAIL  ops={ops}  inputs={inputs}  computed={computed}  expected={expected}")

    print(f"OK:   {n_ok:,}")
    print(f"FAIL: {n_fail:,}")
    print("All correct!" if n_fail == 0 else f"!!! {n_fail} mismatches found !!!")


if __name__ == "__main__":
    main()
