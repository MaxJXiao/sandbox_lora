"""
generate_dataset.py
-------------------
Generates arithmetic datasets for the sandbox_lora experiment.

Modulus: sampled per-row from all 2-digit primes [11, 13, ..., 97] for full generality.
All output is fully reproducible given fixed seeds.

Level 1 — 7 tasks × 2000 samples = 14,000 rows → datasets/level_1.csv
  Columns: task, a, b, answer
  All tasks pooled and shuffled with seed 1.

Level 2 — 49 tasks × 2000 samples = 98,000 rows → datasets/level_2.csv
  Columns: op1, op2, a, b, c, answer
  All pairs pooled and shuffled with seed 2.

Level 3 — 343 tasks × 2000 samples = 686,000 rows → datasets/level_3.csv
  Columns: op1, op2, op3, a, b, c, d, answer
  All triples pooled and shuffled with seed 3.

Level 4 — 2401 tasks × 750 samples = 1,800,750 rows → datasets/level_4.csv
  Columns: op1, op2, op3, op4, a, b, c, d, e, answer
  All quads pooled and shuffled with seed 4.

Level 5 — 16807 tasks × 100 samples = 1,680,700 rows → datasets/level_5.csv
  Columns: op1, op2, op3, op4, op5, a, b, c, d, e, f, answer
  All quints pooled and shuffled with seed 5.

Usage:
  python generate_dataset.py
  python generate_dataset.py --samples 2000 --samples_l4 750 --samples_l5 100 \
      --seed_l1 1 --seed_l2 2 --seed_l3 3 --seed_l4 4 --seed_l5 5
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
# [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def mod_inverse(a, p):
    return pow(a, p - 2, p)


FIELDS = {
    1: ["task", "a", "b", "answer"],
    2: ["op1", "op2", "a", "b", "c", "answer"],
    3: ["op1", "op2", "op3", "a", "b", "c", "d", "answer"],
    4: ["op1", "op2", "op3", "op4", "a", "b", "c", "d", "e", "answer"],
    5: ["op1", "op2", "op3", "op4", "op5", "a", "b", "c", "d", "e", "f", "answer"],
}


def write_csv(rows, path, level):
    rows = list(rows)
    if not rows:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS[level])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows):>7,} rows  →  {path}")


# ---------------------------------------------------------------------------
# Primitive apply functions
# apply_*(x, rng) → (operand_or_dict, answer)
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

def gen_l2_pair(name_i, name_j, apply_j, n, rng):
    for _ in range(n):
        # a, b defaults for non-mod ops; mod ops override inside the y1 block
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
# Level 3 generator — task_i → task_j → task_k
# ---------------------------------------------------------------------------

def gen_l3_triple(name_i, name_j, name_k, apply_j, apply_k, n, rng):
    for row in gen_l2_pair(name_i, name_j, apply_j, n, rng):
        y2 = row.pop("answer")
        new_row = {"op1": row.pop("op1"), "op2": row.pop("op2"), "op3": name_k}
        new_row.update(row)
        c_k, answer = apply_k(y2, rng)
        if isinstance(c_k, tuple):
            new_row["d"], new_row["op3"] = c_k
        else:
            new_row["d"] = c_k
        new_row["answer"] = answer
        yield new_row


# ---------------------------------------------------------------------------
# Level 4 generator — task_i → task_j → task_k → task_l
# ---------------------------------------------------------------------------

def gen_l4_quad(name_i, name_j, name_k, name_l, apply_j, apply_k, apply_l, n, rng):
    for row in gen_l3_triple(name_i, name_j, name_k, apply_j, apply_k, n, rng):
        y3 = row.pop("answer")
        new_row = {
            "op1": row.pop("op1"),
            "op2": row.pop("op2"),
            "op3": row.pop("op3"),
            "op4": name_l,
        }
        new_row.update(row)
        c_l, answer = apply_l(y3, rng)
        if isinstance(c_l, tuple):
            new_row["e"], new_row["op4"] = c_l
        else:
            new_row["e"] = c_l
        new_row["answer"] = answer
        yield new_row


# ---------------------------------------------------------------------------
# Level 5 generator — task_i → task_j → task_k → task_l → task_m
# ---------------------------------------------------------------------------

def gen_l5_quint(name_i, name_j, name_k, name_l, name_m, apply_j, apply_k, apply_l, apply_m, n, rng):
    for row in gen_l4_quad(name_i, name_j, name_k, name_l, apply_j, apply_k, apply_l, n, rng):
        y4 = row.pop("answer")
        new_row = {
            "op1": row.pop("op1"),
            "op2": row.pop("op2"),
            "op3": row.pop("op3"),
            "op4": row.pop("op4"),
            "op5": name_m,
        }
        new_row.update(row)
        c_m, answer = apply_m(y4, rng)
        if isinstance(c_m, tuple):
            new_row["f"], new_row["op5"] = c_m
        else:
            new_row["f"] = c_m
        new_row["answer"] = answer
        yield new_row


# ---------------------------------------------------------------------------
# Registry
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=2000)
    parser.add_argument("--samples_l4", type=int, default=750)
    parser.add_argument("--samples_l5", type=int, default=100)
    parser.add_argument("--seed_l1",    type=int, default=1)
    parser.add_argument("--seed_l2",    type=int, default=2)
    parser.add_argument("--seed_l3",    type=int, default=3)
    parser.add_argument("--seed_l4",    type=int, default=4)
    parser.add_argument("--seed_l5",    type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()

    n          = args.samples
    out        = args.output_dir
    registry   = make_registry()
    task_names = list(registry.keys())

    os.makedirs(out, exist_ok=True)

    # --- Level 1 ---
    rng1 = random.Random(args.seed_l1)
    print(f"\n=== Level 1 | seed {args.seed_l1} | {n:,} samples × {len(task_names)} tasks ===\n")
    l1_rows = []
    for _, (gen_fn, _) in registry.items():
        l1_rows.extend(gen_fn(n, rng1))
    rng1.shuffle(l1_rows)
    write_csv(l1_rows, os.path.join(out, "level_1.csv"), level=1)

    # --- Level 2 ---
    rng2 = random.Random(args.seed_l2)
    print(f"\n=== Level 2 | seed {args.seed_l2} | {n:,} samples × {len(task_names)**2} tasks ===\n")
    l2_rows = []
    for name_i in task_names:
        for name_j in task_names:
            _, apply_j = registry[name_j]
            l2_rows.extend(gen_l2_pair(name_i, name_j, apply_j, n, rng2))
    rng2.shuffle(l2_rows)
    write_csv(l2_rows, os.path.join(out, "level_2.csv"), level=2)

    # --- Level 3 ---
    rng3 = random.Random(args.seed_l3)
    print(f"\n=== Level 3 | seed {args.seed_l3} | {n:,} samples × {len(task_names)**3} tasks ===\n")
    l3_rows = []
    for name_i in task_names:
        for name_j in task_names:
            for name_k in task_names:
                _, apply_j = registry[name_j]
                _, apply_k = registry[name_k]
                l3_rows.extend(gen_l3_triple(name_i, name_j, name_k, apply_j, apply_k, n, rng3))
    rng3.shuffle(l3_rows)
    write_csv(l3_rows, os.path.join(out, "level_3.csv"), level=3)

    # --- Level 4 ---
    rng4 = random.Random(args.seed_l4)
    print(f"\n=== Level 4 | seed {args.seed_l4} | {args.samples_l4:,} samples × {len(task_names)**4} tasks ===\n")
    l4_rows = []
    for name_i in task_names:
        for name_j in task_names:
            for name_k in task_names:
                for name_l in task_names:
                    _, apply_j = registry[name_j]
                    _, apply_k = registry[name_k]
                    _, apply_l = registry[name_l]
                    l4_rows.extend(gen_l4_quad(name_i, name_j, name_k, name_l, apply_j, apply_k, apply_l, args.samples_l4, rng4))
    rng4.shuffle(l4_rows)
    write_csv(l4_rows, os.path.join(out, "level_4.csv"), level=4)

    # --- Level 5 ---
    rng5 = random.Random(args.seed_l5)
    print(f"\n=== Level 5 | seed {args.seed_l5} | {args.samples_l5:,} samples × {len(task_names)**5} tasks ===\n")
    l5_rows = []
    for name_i in task_names:
        for name_j in task_names:
            for name_k in task_names:
                for name_l in task_names:
                    for name_m in task_names:
                        _, apply_j = registry[name_j]
                        _, apply_k = registry[name_k]
                        _, apply_l = registry[name_l]
                        _, apply_m = registry[name_m]
                        l5_rows.extend(gen_l5_quint(name_i, name_j, name_k, name_l, name_m, apply_j, apply_k, apply_l, apply_m, args.samples_l5, rng5))
    rng5.shuffle(l5_rows)
    write_csv(l5_rows, os.path.join(out, "level_5.csv"), level=5)

    print("\nDone.\n")


if __name__ == "__main__":
    main()