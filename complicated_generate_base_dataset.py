"""
complicated_generate_dataset_base.py (unfinished)
------------------------
Generates arithmetic datasets for the sandbox_lora experiment — base-converted variant.

Same 7 operations and compositional staircase as generate_dataset.py, but every
number (operands, answers, AND the modulus in op names) is expressed as a digit
string in a randomly sampled base (7–16).

Train/val/test splitting is done downstream using the base column:
  Training   — bases 7–14
  Val/Test   — bases 15–16 (digits E=14, F=15 never seen in training)

Digit symbol table (hex-style):
  0–9 → 0 1 2 3 4 5 6 7 8 9
  10→A  11→B  12→C  13→D  14→E  15→F

7 operations (same as generate_dataset.py):
  mod_add, mod_sub, mod_mul, mod_div — modular arithmetic, prime modulus
  2D+, 2D-, 2Dx                     — two-digit add/sub/mul

CSV schema — ALL numeric values are base-b digit strings:
  Level 1: base, task, a, b, answer
  Level 2: base, op1, op2, a, b, c, answer
  Level 3: base, op1, op2, op3, a, b, c, d, answer
  Level 4: base, op1, op2, op3, op4, a, b, c, d, e, answer
  Level 5: base, op1, op2, op3, op4, op5, a, b, c, d, e, f, answer

  task / op names encode the modulus in-base, e.g. "mod_+_B2" means
  mod-add with prime 97 when base is 9 (since 97 decimal = B2 in base 9).
  2D ops keep their original names: "2D+", "2D-", "2Dx".

Level 1 — 7 tasks × 2000 samples = 14,000 rows
Level 2 — 49 tasks × 2000 samples = 98,000 rows
Level 3 — 343 tasks × 2000 samples = 686,000 rows
Level 4 — 2401 tasks × 750 samples = 1,800,750 rows
Level 5 — 16807 tasks × 100 samples = 1,680,700 rows

Usage:
  python generate_dataset_base.py
  python generate_dataset_base.py --samples 2000 --samples_l4 750 --samples_l5 100 \\
      --seed_l1 101 --seed_l2 102 --seed_l3 103 --seed_l4 104 --seed_l5 105
"""

import argparse
import csv
import math
import os
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


PRIMES = [p for p in range(10, 100) if is_prime(p)]
# [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def mod_inverse(a, p):
    return pow(a, p - 2, p)


# ---------------------------------------------------------------------------
# Digit symbol table  (0–15, hex-style)
# ---------------------------------------------------------------------------

_DIGIT_SYMBOLS = "0123456789ABCDEF"

assert len(_DIGIT_SYMBOLS) == 16


def int_to_base(n: int, base: int) -> str:
    """Integer → minimal digit string in `base`.  Handles negatives."""
    if n < 0:
        return "-" + int_to_base(-n, base)
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append(_DIGIT_SYMBOLS[n % base])
        n //= base
    return "".join(reversed(digits))


def base_to_int(s: str, base: int) -> int:
    """Digit string in `base` → int.  Handles negatives."""
    if s.startswith("-"):
        return -base_to_int(s[1:], base)
    result = 0
    for ch in s:
        result = result * base + _DIGIT_SYMBOLS.index(ch)
    return result


# ---------------------------------------------------------------------------
# All bases in one pool — split downstream via the base column
# ---------------------------------------------------------------------------

ALL_BASES = list(range(7, 17))   # 7 8 9 10 11 12 13 14 15 16


# ---------------------------------------------------------------------------
# CSV field lists  (same as generate_dataset.py, plus "base" column)
# ---------------------------------------------------------------------------

FIELDS = {
    1: ["base", "task", "a", "b", "answer"],
    2: ["base", "op1", "op2", "a", "b", "c", "answer"],
    3: ["base", "op1", "op2", "op3", "a", "b", "c", "d", "answer"],
    4: ["base", "op1", "op2", "op3", "op4", "a", "b", "c", "d", "e", "answer"],
    5: ["base", "op1", "op2", "op3", "op4", "op5", "a", "b", "c", "d", "e", "f", "answer"],
}


def write_csv(rows, path, level):
    rows = list(rows)
    if not rows:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS[level])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows):>9,} rows  →  {path}")


# ---------------------------------------------------------------------------
# Operation registry — 7 ops, same as generate_dataset.py
# ---------------------------------------------------------------------------

OP_KEYS = ["mod_add", "mod_sub", "mod_mul", "mod_div", "2D+", "2D-", "2Dx"]


# ---------------------------------------------------------------------------
# Position-1 sampler:  sample both operands, compute result
# Returns (y1_int, op_display, a_str, b_str)
# Arithmetic in Python ints; conversion to base at the boundary.
# ---------------------------------------------------------------------------

def sample_first_op(op_key, rng, base):
    if op_key == "mod_add":
        p = rng.choice(PRIMES)
        a = rng.randrange(0, p)
        b = rng.randrange(0, p)
        y1 = (a + b) % p
        op_disp = f"mod_+_{int_to_base(p, base)}"
    elif op_key == "mod_sub":
        p = rng.choice(PRIMES)
        a = rng.randrange(0, p)
        b = rng.randrange(0, p)
        y1 = (a - b) % p
        op_disp = f"mod_-_{int_to_base(p, base)}"
    elif op_key == "mod_mul":
        p = rng.choice(PRIMES)
        a = rng.randrange(0, p)
        b = rng.randrange(0, p)
        y1 = (a * b) % p
        op_disp = f"mod_x_{int_to_base(p, base)}"
    elif op_key == "mod_div":
        p = rng.choice(PRIMES)
        a = rng.randrange(0, p)
        b = rng.randrange(1, p)
        y1 = (a * mod_inverse(b, p)) % p
        op_disp = f"mod_div_{int_to_base(p, base)}"
    elif op_key == "2D+":
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        y1 = a + b
        op_disp = "2D+"
    elif op_key == "2D-":
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        y1 = a - b
        op_disp = "2D-"
    elif op_key == "2Dx":
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        y1 = a * b
        op_disp = "2Dx"

    return y1, op_disp, int_to_base(a, base), int_to_base(b, base)


# ---------------------------------------------------------------------------
# Position 2+ applier:  given previous result (int), sample one operand
# Returns (c_str, op_display, result_int)
# ---------------------------------------------------------------------------

def apply_op(op_key, x_int, rng, base):
    if op_key == "mod_add":
        p = rng.choice(PRIMES)
        c = rng.randrange(0, p)
        return int_to_base(c, base), f"mod_+_{int_to_base(p, base)}", (x_int + c) % p
    elif op_key == "mod_sub":
        p = rng.choice(PRIMES)
        c = rng.randrange(0, p)
        return int_to_base(c, base), f"mod_-_{int_to_base(p, base)}", (x_int - c) % p
    elif op_key == "mod_mul":
        p = rng.choice(PRIMES)
        c = rng.randrange(0, p)
        return int_to_base(c, base), f"mod_x_{int_to_base(p, base)}", (x_int * c) % p
    elif op_key == "mod_div":
        p = rng.choice(PRIMES)
        c = rng.randrange(1, p)
        return int_to_base(c, base), f"mod_div_{int_to_base(p, base)}", (x_int * mod_inverse(c, p)) % p
    elif op_key == "2D+":
        c = rng.randrange(0, 100)
        return int_to_base(c, base), "2D+", x_int + c
    elif op_key == "2D-":
        c = rng.randrange(0, 100)
        return int_to_base(c, base), "2D-", x_int - c
    elif op_key == "2Dx":
        c = rng.randrange(0, 100)
        return int_to_base(c, base), "2Dx", x_int * c


# ---------------------------------------------------------------------------
# Row generators — standalone per level, arithmetic in ints, convert at boundary
# ---------------------------------------------------------------------------

def gen_l1(op_key, n, bases, rng):
    for _ in range(n):
        base = rng.choice(bases)
        y1, task, a, b = sample_first_op(op_key, rng, base)
        yield {"base": base, "task": task, "a": a, "b": b,
               "answer": int_to_base(y1, base)}


def gen_l2(oi, oj, n, bases, rng):
    for _ in range(n):
        base = rng.choice(bases)
        y1, op1, a, b = sample_first_op(oi, rng, base)
        c, op2, ans   = apply_op(oj, y1, rng, base)
        yield {"base": base, "op1": op1, "op2": op2,
               "a": a, "b": b, "c": c,
               "answer": int_to_base(ans, base)}


def gen_l3(oi, oj, ok, n, bases, rng):
    for _ in range(n):
        base = rng.choice(bases)
        y1, op1, a, b = sample_first_op(oi, rng, base)
        c, op2, y2    = apply_op(oj, y1, rng, base)
        d, op3, ans   = apply_op(ok, y2, rng, base)
        yield {"base": base, "op1": op1, "op2": op2, "op3": op3,
               "a": a, "b": b, "c": c, "d": d,
               "answer": int_to_base(ans, base)}


def gen_l4(oi, oj, ok, ol, n, bases, rng):
    for _ in range(n):
        base = rng.choice(bases)
        y1, op1, a, b = sample_first_op(oi, rng, base)
        c, op2, y2    = apply_op(oj, y1, rng, base)
        d, op3, y3    = apply_op(ok, y2, rng, base)
        e, op4, ans   = apply_op(ol, y3, rng, base)
        yield {"base": base, "op1": op1, "op2": op2, "op3": op3, "op4": op4,
               "a": a, "b": b, "c": c, "d": d, "e": e,
               "answer": int_to_base(ans, base)}


def gen_l5(oi, oj, ok, ol, om, n, bases, rng):
    for _ in range(n):
        base = rng.choice(bases)
        y1, op1, a, b = sample_first_op(oi, rng, base)
        c, op2, y2    = apply_op(oj, y1, rng, base)
        d, op3, y3    = apply_op(ok, y2, rng, base)
        e, op4, y4    = apply_op(ol, y3, rng, base)
        f, op5, ans   = apply_op(om, y4, rng, base)
        yield {"base": base, "op1": op1, "op2": op2, "op3": op3, "op4": op4, "op5": op5,
               "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
               "answer": int_to_base(ans, base)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=2000)
    parser.add_argument("--samples_l4", type=int, default=750)
    parser.add_argument("--samples_l5", type=int, default=100)
    parser.add_argument("--seed_l1",    type=int, default=101)
    parser.add_argument("--seed_l2",    type=int, default=102)
    parser.add_argument("--seed_l3",    type=int, default=103)
    parser.add_argument("--seed_l4",    type=int, default=104)
    parser.add_argument("--seed_l5",    type=int, default=105)
    parser.add_argument("--output_dir", type=str, default="complicated_base_datasets")
    args = parser.parse_args()

    n   = args.samples
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # --- Level 1 ---
    rng1 = random.Random(args.seed_l1)
    print(f"\n=== Level 1 | seed {args.seed_l1} | {n:,} samples × {len(OP_KEYS)} tasks ===\n")
    l1_rows = []
    for op in OP_KEYS:
        l1_rows.extend(gen_l1(op, n, ALL_BASES, rng1))
    rng1.shuffle(l1_rows)
    write_csv(l1_rows, os.path.join(out, "level_1.csv"), level=1)

    # --- Level 2 ---
    rng2 = random.Random(args.seed_l2)
    print(f"\n=== Level 2 | seed {args.seed_l2} | {n:,} samples × {len(OP_KEYS)**2} tasks ===\n")
    l2_rows = []
    for oi in OP_KEYS:
        for oj in OP_KEYS:
            l2_rows.extend(gen_l2(oi, oj, n, ALL_BASES, rng2))
    rng2.shuffle(l2_rows)
    write_csv(l2_rows, os.path.join(out, "level_2.csv"), level=2)

    # --- Level 3 ---
    rng3 = random.Random(args.seed_l3)
    print(f"\n=== Level 3 | seed {args.seed_l3} | {n:,} samples × {len(OP_KEYS)**3} tasks ===\n")
    l3_rows = []
    for oi in OP_KEYS:
        for oj in OP_KEYS:
            for ok in OP_KEYS:
                l3_rows.extend(gen_l3(oi, oj, ok, n, ALL_BASES, rng3))
    rng3.shuffle(l3_rows)
    write_csv(l3_rows, os.path.join(out, "level_3.csv"), level=3)

    # --- Level 4 ---
    rng4 = random.Random(args.seed_l4)
    print(f"\n=== Level 4 | seed {args.seed_l4} | {args.samples_l4:,} samples × {len(OP_KEYS)**4} tasks ===\n")
    l4_rows = []
    for oi in OP_KEYS:
        for oj in OP_KEYS:
            for ok in OP_KEYS:
                for ol in OP_KEYS:
                    l4_rows.extend(gen_l4(oi, oj, ok, ol, args.samples_l4, ALL_BASES, rng4))
    rng4.shuffle(l4_rows)
    write_csv(l4_rows, os.path.join(out, "level_4.csv"), level=4)

    # --- Level 5 ---
    rng5 = random.Random(args.seed_l5)
    print(f"\n=== Level 5 | seed {args.seed_l5} | {args.samples_l5:,} samples × {len(OP_KEYS)**5} tasks ===\n")
    l5_rows = []
    for oi in OP_KEYS:
        for oj in OP_KEYS:
            for ok in OP_KEYS:
                for ol in OP_KEYS:
                    for om in OP_KEYS:
                        l5_rows.extend(gen_l5(oi, oj, ok, ol, om, args.samples_l5, ALL_BASES, rng5))
    rng5.shuffle(l5_rows)
    write_csv(l5_rows, os.path.join(out, "level_5.csv"), level=5)

    print("\nDone.\n")


if __name__ == "__main__":
    main()