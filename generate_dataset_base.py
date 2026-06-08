"""
generate_dataset_base.py
------------------------
Generates base-arithmetic datasets for the sandbox_lora OOD experiment.

All arithmetic is performed in a randomly sampled base (7–16).
Every operand and answer is expressed as a digit string in that base.
The base is recorded as a plain integer in its own column.

Train/val/test splitting is done downstream using the base column:
  Training   — bases 7–14
  Val/Test   — bases 15–16 (digits E=14 and F=15 never seen in training)

Digit symbol table (hex-style):
  0–9 → 0 1 2 3 4 5 6 7 8 9
  10→A  11→B  12→C  13→D  14→E  15→F

Operations (3 primitives, closed via mod b^width):
  add — (a + b) mod b^width
  sub — (a - b) mod b^width
  mul — (a * b) mod b^width

CSV schema — operands and answer are base-b digit strings:
  Level 1: base, task,                      a, b,                answer
  Level 2: base, op1, op2,                  a, b, c,             answer
  Level 3: base, op1, op2, op3,             a, b, c, d,          answer
  Level 4: base, op1, op2, op3, op4,        a, b, c, d, e,       answer
  Level 5: base, op1, op2, op3, op4, op5,   a, b, c, d, e, f,   answer

Per-level sample counts (samples per op-tuple × 3^level op-tuples):
  Level 1 — 5000 × 3   =  15,000 rows → base_datasets/level_1.csv
  Level 2 — 3000 × 9   =  27,000 rows → base_datasets/level_2.csv
  Level 3 — 3000 × 27  =  81,000 rows → base_datasets/level_3.csv
  Level 4 — 1000 × 81  =  81,000 rows → base_datasets/level_4.csv
  Level 5 —  300 × 243 =  72,900 rows → base_datasets/level_5.csv

Usage:
  python generate_dataset_base.py
  python generate_dataset_base.py --max_digits 2 --samples_l1 2000 \\
      --samples_l5 300 --max_level 5 --output_dir base_datasets
"""

import argparse
import csv
import os
import random

# ---------------------------------------------------------------------------
# Digit symbol table  (0–15, hex-style)
# ---------------------------------------------------------------------------

_DIGIT_SYMBOLS = "0123456789ABCDEF"

assert len(_DIGIT_SYMBOLS) == 16


def int_to_base(n: int, base: int, width: int) -> str:
    """Non-negative int → zero-padded digit string in `base`."""
    assert 2 <= base <= 16
    if n == 0:
        return _DIGIT_SYMBOLS[0] * width
    digits = []
    while n > 0:
        digits.append(_DIGIT_SYMBOLS[n % base])
        n //= base
    while len(digits) < width:
        digits.append(_DIGIT_SYMBOLS[0])
    return "".join(reversed(digits))


def base_to_int(s: str, base: int) -> int:
    """Digit string in `base` → int."""
    result = 0
    for ch in s:
        result = result * base + _DIGIT_SYMBOLS.index(ch)
    return result


# ---------------------------------------------------------------------------
# Arithmetic (modular, stays within `width` digits)
# ---------------------------------------------------------------------------

def _mod(n, base, width):
    return n % (base ** width)

def arith_add(x, y, base, width): return _mod(x + y, base, width)
def arith_sub(x, y, base, width): return _mod(x - y, base, width)
def arith_mul(x, y, base, width): return _mod(x * y, base, width)

_ARITH = {"add": arith_add, "sub": arith_sub, "mul": arith_mul}

OP_KEYS = list(_ARITH.keys())   # ["add", "sub", "mul"]


# ---------------------------------------------------------------------------
# Apply: given x (int), sample y, return (y_str, result_int)
# ---------------------------------------------------------------------------

def apply_op(op_key: str, x_int: int, base: int, width: int, rng: random.Random):
    limit = base ** width
    y_int = rng.randrange(0, limit)
    result = _ARITH[op_key](x_int, y_int, base, width)
    return int_to_base(y_int, base, width), result


# ---------------------------------------------------------------------------
# All bases in one pool — split downstream via the base column
# ---------------------------------------------------------------------------

ALL_BASES = list(range(7, 17))   # 7 8 9 10 11 12 13 14 15 16


# ---------------------------------------------------------------------------
# CSV field lists
# ---------------------------------------------------------------------------

FIELDS = {
    1: ["base", "task",                                  "a", "b",                         "answer"],
    2: ["base", "op1", "op2",                            "a", "b", "c",                    "answer"],
    3: ["base", "op1", "op2", "op3",                     "a", "b", "c", "d",               "answer"],
    4: ["base", "op1", "op2", "op3", "op4",              "a", "b", "c", "d", "e",          "answer"],
    5: ["base", "op1", "op2", "op3", "op4", "op5",       "a", "b", "c", "d", "e", "f",    "answer"],
}


# ---------------------------------------------------------------------------
# Row generators
# ---------------------------------------------------------------------------

def gen_l1(op_key, n, bases, width, rng):
    for _ in range(n):
        base  = rng.choice(bases)
        a_int = rng.randrange(0, base ** width)
        a_str = int_to_base(a_int, base, width)
        b_str, ans_int = apply_op(op_key, a_int, base, width, rng)
        yield {
            "base": base, "task": op_key,
            "a": a_str, "b": b_str,
            "answer": int_to_base(ans_int, base, width),
        }


def gen_l2(oi, oj, n, bases, width, rng):
    for _ in range(n):
        base  = rng.choice(bases)
        a_int = rng.randrange(0, base ** width)
        a_str = int_to_base(a_int, base, width)

        b_str, y1  = apply_op(oi, a_int, base, width, rng)
        c_str, ans = apply_op(oj, y1,    base, width, rng)

        yield {
            "base": base, "op1": oi, "op2": oj,
            "a": a_str, "b": b_str, "c": c_str,
            "answer": int_to_base(ans, base, width),
        }


def gen_l3(oi, oj, ok, n, bases, width, rng):
    for _ in range(n):
        base  = rng.choice(bases)
        a_int = rng.randrange(0, base ** width)
        a_str = int_to_base(a_int, base, width)

        b_str, y1  = apply_op(oi, a_int, base, width, rng)
        c_str, y2  = apply_op(oj, y1,    base, width, rng)
        d_str, ans = apply_op(ok, y2,    base, width, rng)

        yield {
            "base": base, "op1": oi, "op2": oj, "op3": ok,
            "a": a_str, "b": b_str, "c": c_str, "d": d_str,
            "answer": int_to_base(ans, base, width),
        }


def gen_l4(oi, oj, ok, ol, n, bases, width, rng):
    for _ in range(n):
        base  = rng.choice(bases)
        a_int = rng.randrange(0, base ** width)
        a_str = int_to_base(a_int, base, width)

        b_str, y1  = apply_op(oi, a_int, base, width, rng)
        c_str, y2  = apply_op(oj, y1,    base, width, rng)
        d_str, y3  = apply_op(ok, y2,    base, width, rng)
        e_str, ans = apply_op(ol, y3,    base, width, rng)

        yield {
            "base": base, "op1": oi, "op2": oj, "op3": ok, "op4": ol,
            "a": a_str, "b": b_str, "c": c_str, "d": d_str, "e": e_str,
            "answer": int_to_base(ans, base, width),
        }


def gen_l5(oi, oj, ok, ol, om, n, bases, width, rng):
    for _ in range(n):
        base  = rng.choice(bases)
        a_int = rng.randrange(0, base ** width)
        a_str = int_to_base(a_int, base, width)

        b_str, y1  = apply_op(oi, a_int, base, width, rng)
        c_str, y2  = apply_op(oj, y1,    base, width, rng)
        d_str, y3  = apply_op(ok, y2,    base, width, rng)
        e_str, y4  = apply_op(ol, y3,    base, width, rng)
        f_str, ans = apply_op(om, y4,    base, width, rng)

        yield {
            "base": base, "op1": oi, "op2": oj, "op3": ok, "op4": ol, "op5": om,
            "a": a_str, "b": b_str, "c": c_str, "d": d_str, "e": e_str, "f": f_str,
            "answer": int_to_base(ans, base, width),
        }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows, path, level):
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS[level])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows):>9,} rows  →  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_digits",  type=int, default=2)
    parser.add_argument("--samples_l1",  type=int, default=5000,
                        help="Samples per op-tuple, level 1")
    parser.add_argument("--samples_l2",  type=int, default=3000)
    parser.add_argument("--samples_l3",  type=int, default=3000)
    parser.add_argument("--samples_l4",  type=int, default=1000)
    parser.add_argument("--samples_l5",  type=int, default=300)
    parser.add_argument("--max_level",   type=int, default=5)
    parser.add_argument("--seed_l1",     type=int, default=101)
    parser.add_argument("--seed_l2",     type=int, default=102)
    parser.add_argument("--seed_l3",     type=int, default=103)
    parser.add_argument("--seed_l4",     type=int, default=104)
    parser.add_argument("--seed_l5",     type=int, default=105)
    parser.add_argument("--output_dir",  type=str, default="base_datasets")
    args = parser.parse_args()

    width  = args.max_digits
    out    = args.output_dir
    os.makedirs(out, exist_ok=True)

    seeds     = {1: args.seed_l1, 2: args.seed_l2, 3: args.seed_l3,
                 4: args.seed_l4, 5: args.seed_l5}
    n_per_lvl = {1: args.samples_l1, 2: args.samples_l2, 3: args.samples_l3,
                 4: args.samples_l4, 5: args.samples_l5}
    op_count  = {l: len(OP_KEYS)**l for l in range(1, 6)}

    for level in range(1, args.max_level + 1):
        n   = n_per_lvl[level]
        rng = random.Random(seeds[level])
        rows = []

        print(f"\n=== Level {level} | bases {ALL_BASES} | seed {seeds[level]} "
              f"| {n:,} samples × {op_count[level]} op-tuples ===\n")

        if level == 1:
            for oi in OP_KEYS:
                rows.extend(gen_l1(oi, n, ALL_BASES, width, rng))
        elif level == 2:
            for oi in OP_KEYS:
                for oj in OP_KEYS:
                    rows.extend(gen_l2(oi, oj, n, ALL_BASES, width, rng))
        elif level == 3:
            for oi in OP_KEYS:
                for oj in OP_KEYS:
                    for ok in OP_KEYS:
                        rows.extend(gen_l3(oi, oj, ok, n, ALL_BASES, width, rng))
        elif level == 4:
            for oi in OP_KEYS:
                for oj in OP_KEYS:
                    for ok in OP_KEYS:
                        for ol in OP_KEYS:
                            rows.extend(gen_l4(oi, oj, ok, ol, n, ALL_BASES, width, rng))
        elif level == 5:
            for oi in OP_KEYS:
                for oj in OP_KEYS:
                    for ok in OP_KEYS:
                        for ol in OP_KEYS:
                            for om in OP_KEYS:
                                rows.extend(gen_l5(oi, oj, ok, ol, om, n, ALL_BASES, width, rng))

        rng.shuffle(rows)
        write_csv(rows, os.path.join(out, f"level_{level}.csv"), level=level)

    print("\nDone.\n")


if __name__ == "__main__":
    main()