"""
simple_dataset.py
-----------------
Generates per-task CSVs for individual task evaluation (trial runs).

Modulus: sampled per-row from all 2-digit primes [11, 13, ..., 97] for full generality.
All output is fully reproducible given fixed seeds.

Level 1 — 7 tasks × 2000 samples each → trial_dataset/level_1/<task>.csv
  mod_add, mod_sub, mod_mul, mod_div  →  (a op b) mod p, p sampled per row
  2D+, 2D-                            →  a,b in [0, 100)
  2Dx                                 →  a,b in [0, 100)
  Seed: 1

Level 2 — 49 tasks × 2000 samples each → trial_dataset/level_2/<task_i>__<task_j>.csv
  Every Level 1 task composed with every other: answer = task_j(task_i(a, b), c)
  Columns: op1, op2, a, b, c, answer
  Seed: 2

Level 3 — 343 tasks × 2000 samples each → trial_dataset/level_3/<task_i>__<task_j>__<task_k>.csv
  Three-step chain: answer = task_k(task_j(task_i(a, b), c), d)
  Columns: op1, op2, op3, a, b, c, d, answer
  Seed: 3

Usage:
  python simple_dataset.py
  python simple_dataset.py --samples 2000 --seed_l1 1 --seed_l2 2 --seed_l3 3
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


# Primes between 10 and 99 — modulus sampled randomly from this list each row
PRIMES = [p for p in range(10, 100) if is_prime(p)]


def mod_inverse(a, p):
    return pow(a, p - 2, p)


FIELDS = {
    1: ["task", "a", "b", "answer"],
    2: ["op1", "op2", "a", "b", "c", "answer"],
    3: ["op1", "op2", "op3", "a", "b", "c", "d", "answer"],
}


def write_csv(rows, path, level):
    rows = list(rows)
    if not rows:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS[level], restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows):>7,} rows  →  {path}")


# ---------------------------------------------------------------------------
# Primitive apply functions
# apply_*(x, rng) → (operand_or_tuple, answer)
# Mod ops return (c, label_str) tuple; 2D ops return plain c.
# ---------------------------------------------------------------------------

def apply_mod_add(x, rng):
    p = rng.choice(PRIMES)
    c = rng.randrange(0, p)
    return (c, f"mod_+_{p}"), (x + c) % p

def apply_mod_sub(x, rng):
    p = rng.choice(PRIMES)
    c = rng.randrange(0, p)
    return (c, f"mod_-_{p}"), (x - c) % p

def apply_mod_mul(x, rng):
    p = rng.choice(PRIMES)
    c = rng.randrange(0, p)
    return (c, f"mod_x_{p}"), (x * c) % p

def apply_mod_div(x, rng):
    p = rng.choice(PRIMES)
    c = rng.randrange(1, p)
    return (c, f"mod_div_{p}"), (x * mod_inverse(c, p)) % p

def apply_2d_add(x, rng):
    c = rng.randrange(0, 100)
    return c, x + c

def apply_2d_sub(x, rng):
    c = rng.randrange(0, 100)
    return c, x - c

def apply_2d_mul(x, rng):
    c = rng.randrange(0, 100)
    return c, x * c


# ---------------------------------------------------------------------------
# Level 1 generators
# ---------------------------------------------------------------------------

def gen_l1_mod_op(op, n, rng):
    for _ in range(n):
        p = rng.choice(PRIMES)
        a = rng.randrange(0, p)
        b = rng.randrange(1 if op == "/" else 0, p)
        if op == "+":
            answer = (a + b) % p
        elif op == "-":
            answer = (a - b) % p
        elif op == "*":
            answer = (a * b) % p
        elif op == "/":
            answer = (a * mod_inverse(b, p)) % p
        task = f"mod_{'+' if op == '+' else '-' if op == '-' else 'x' if op == '*' else 'div'}_{p}"
        yield {"task": task, "a": a, "b": b, "answer": answer}

def gen_l1_2d_add(n, rng):
    for _ in range(n):
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        yield {"task": "2D+", "a": a, "b": b, "answer": a + b}

def gen_l1_2d_sub(n, rng):
    for _ in range(n):
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        yield {"task": "2D-", "a": a, "b": b, "answer": a - b}

def gen_l1_2d_mul(n, rng):
    for _ in range(n):
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        yield {"task": "2Dx", "a": a, "b": b, "answer": a * b}


# ---------------------------------------------------------------------------
# Task registry
# Maps task name → (level1_generator_fn, apply_fn)
# ---------------------------------------------------------------------------

def make_registry():
    return {
        "mod_add": (
            lambda n, rng: gen_l1_mod_op("+", n, rng),
            lambda x, rng: apply_mod_add(x, rng),
        ),
        "mod_sub": (
            lambda n, rng: gen_l1_mod_op("-", n, rng),
            lambda x, rng: apply_mod_sub(x, rng),
        ),
        "mod_mul": (
            lambda n, rng: gen_l1_mod_op("*", n, rng),
            lambda x, rng: apply_mod_mul(x, rng),
        ),
        "mod_div": (
            lambda n, rng: gen_l1_mod_op("/", n, rng),
            lambda x, rng: apply_mod_div(x, rng),
        ),
        "2D+": (
            lambda n, rng: gen_l1_2d_add(n, rng),
            lambda x, rng: apply_2d_add(x, rng),
        ),
        "2D-": (
            lambda n, rng: gen_l1_2d_sub(n, rng),
            lambda x, rng: apply_2d_sub(x, rng),
        ),
        "2Dx": (
            lambda n, rng: gen_l1_2d_mul(n, rng),
            lambda x, rng: apply_2d_mul(x, rng),
        ),
    }


def gen_l2_pair(name_i, name_j, apply_j, n, rng):
    for _ in range(n):
        a = rng.randrange(0, 100) if name_i in ("2D+", "2D-", "2Dx") else 0
        b_hi = 100 if name_i in ("2D+", "2D-", "2Dx") else 0
        b = rng.randrange(0, b_hi) if b_hi > 0 else 0

        row = {"op1": name_i, "op2": name_j, "a": a, "b": b}

        if name_i == "mod_add":
            p_i = rng.choice(PRIMES)
            a = rng.randrange(0, p_i)
            b = rng.randrange(0, p_i)
            y1 = (a + b) % p_i
            row["op1"] = f"mod_+_{p_i}"
            row["a"] = a
            row["b"] = b
        elif name_i == "mod_sub":
            p_i = rng.choice(PRIMES)
            a = rng.randrange(0, p_i)
            b = rng.randrange(0, p_i)
            y1 = (a - b) % p_i
            row["op1"] = f"mod_-_{p_i}"
            row["a"] = a
            row["b"] = b
        elif name_i == "mod_mul":
            p_i = rng.choice(PRIMES)
            a = rng.randrange(0, p_i)
            b = rng.randrange(0, p_i)
            y1 = (a * b) % p_i
            row["op1"] = f"mod_x_{p_i}"
            row["a"] = a
            row["b"] = b
        elif name_i == "mod_div":
            p_i = rng.choice(PRIMES)
            a = rng.randrange(0, p_i)
            b = rng.randrange(1, p_i)
            y1 = (a * mod_inverse(b, p_i)) % p_i
            row["op1"] = f"mod_div_{p_i}"
            row["a"] = a
            row["b"] = b
        elif name_i == "2D+":
            y1 = a + b
        elif name_i == "2D-":
            y1 = a - b
        elif name_i == "2Dx":
            y1 = a * b

        c_j, answer = apply_j(y1, rng)
        if isinstance(c_j, tuple):
            row["c"], row["op2"] = c_j
        else:
            row["c"] = c_j

        row["answer"] = answer
        yield row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=2_000)
    parser.add_argument("--seed_l1",    type=int, default=1)
    parser.add_argument("--seed_l2",    type=int, default=2)
    parser.add_argument("--seed_l3",    type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="trial_dataset")
    args = parser.parse_args()

    n   = args.samples
    out = args.output_dir
    registry = make_registry()
    task_names = list(registry.keys())

    # --- Level 1 ---
    rng1 = random.Random(args.seed_l1)
    l1_out = os.path.join(out, "level_1")
    print(f"\n=== Level 1 | seed {args.seed_l1} | {n:,} samples each | {len(task_names)} tasks ===\n")

    for name, (gen_fn, _) in registry.items():
        write_csv(gen_fn(n, rng1), os.path.join(l1_out, f"{name}.csv"), level=1)

    # --- Level 2 ---
    rng2 = random.Random(args.seed_l2)
    l2_out = os.path.join(out, "level_2")
    print(f"\n=== Level 2 | seed {args.seed_l2} | {n:,} samples each | {len(task_names)**2} tasks ===\n")

    for name_i in task_names:
        for name_j in task_names:
            _, apply_j = registry[name_j]
            fname = f"{name_i}__{name_j}.csv"
            write_csv(
                gen_l2_pair(name_i, name_j, apply_j, n, rng2),
                os.path.join(l2_out, fname),
                level=2,
            )

    # --- Level 3 ---
    rng3 = random.Random(args.seed_l3)
    l3_out = os.path.join(out, "level_3")
    print(f"\n=== Level 3 | seed {args.seed_l3} | {n:,} samples each | {len(task_names)**3} tasks ===\n")

    for name_i in task_names:
        for name_j in task_names:
            for name_k in task_names:
                _, apply_j = registry[name_j]
                _, apply_k = registry[name_k]
                fname = f"{name_i}__{name_j}__{name_k}.csv"

                def gen_l3(name_i=name_i, name_j=name_j, name_k=name_k, apply_j=apply_j, apply_k=apply_k):
                    for row in gen_l2_pair(name_i, name_j, apply_j, n, rng3):
                        y2 = row["answer"]
                        row["op3"] = name_k
                        c_k, answer = apply_k(y2, rng3)
                        if isinstance(c_k, tuple):
                            row["d"], row["op3"] = c_k
                        else:
                            row["d"] = c_k
                        row["answer"] = answer
                        yield row

                write_csv(gen_l3(), os.path.join(l3_out, fname), level=3)

    total = len(task_names) + len(task_names)**2 + len(task_names)**3
    print(f"\nDone. {total} files total.\n")


if __name__ == "__main__":
    main()
