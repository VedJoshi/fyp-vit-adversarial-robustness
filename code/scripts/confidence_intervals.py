"""95% Wilson intervals for the reported figures, and paired tests between them.

    python code/scripts/confidence_intervals.py [code/results]

Every figure is a count of images out of n, read from its saved record. The intervals cover
image sampling only; robust accuracies remain upper bounds on attack strength.

Paired comparisons use McNemar's exact test. Where both records list which images were correct,
p is exact. Otherwise it is the largest p consistent with the two totals on the same images, so
only a small bound shows a difference; a large bound shows nothing.
"""
import json
import math
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "code/results")
Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_cdf(k, m, p=0.5):
    if p == 0.5:
        return sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(k + 1))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p from the two discordant counts."""
    return min(1.0, 2 * binom_cdf(min(b, c), b + c)) if b + c else 1.0


def mcnemar_bound(k1, k2, n):
    """Largest exact two-sided McNemar p over pairings of k1 and k2 successes in n."""
    hi, lo = max(k1, k2), min(k1, k2)
    c = min(lo, n - hi)            # p grows with c at fixed b - c, so take the most c allows
    return mcnemar_exact(c + hi - lo, c)


def get(path, *keys):
    x = json.loads((R / path).read_text(encoding="utf-8"))
    for k in keys:
        x = x[k]
    return x


def count(pct, n):
    k = round(pct * n / 100)
    assert abs(100 * k / n - pct) < 1e-9, (pct, n)
    return k


def union(*lists):
    return len(set().union(*map(set, lists)))


CORRECT: dict[str, set[int]] = {}       # per-image correct sets, where a record keeps them


def figures():
    """(key, label, k, n, status). status: 'established' or 'preliminary'."""
    out = []

    def pct(key, label, n, status, path, *keys):
        out.append((key, label, count(get(path, *keys), n), n, status))

    E, P = "established", "preliminary"
    pct("released_all", "RSA, all released settings", 512, E,
        "gate_a_released.json", "per_size", "20", "robust_acc_pct")
    pct("released_grad", "RSA, released gradient", 512, E,
        "gate_a_bwdrel.json", "per_size", "20", "robust_acc_pct")
    pct("correct_grad", "RSA, correct gradient", 512, E,
        "gate_a_reference.json", "per_size", "20", "robust_acc_pct")
    pct("oracle", "Oracle: replacement at the true location", 512, E,
        "gate_a_orcfull_oracle_window.json", "per_size", "20", "robust_acc_pct")
    pct("cf_usual", "Centred-Frozen RSA, usual attack", 512, E,
        "gate_a_diag_centred_frz0.json", "per_size", "20", "robust_acc_pct")
    pct("cf_selection", "Centred-Frozen RSA, selection-aware attack", 512, E,
        "gate_a_attack_suite.json", "by_source", "selection", "alone", "worst_case_pct")
    pct("cf_all", "Centred-Frozen RSA, all attacks combined", 512, E,
        "gate_a_attack_suite.json", "suite_union", "worst_case_pct")
    pct("transfer_cf", "Centred-Frozen RSA, transferred patches only", 512, E,
        "gate_a_transfer_suite_arm5_blackbox.json", "transfer_only_union", "worst_case_pct")
    pct("transfer_ref", "Reference, transferred patches only", 512, E,
        "gate_a_transfer_suite_ref.json", "transfer_only_union", "worst_case_pct")
    pct("knnfar1", "Neighbour-distance rule (dev)", 128, E,
        "knnfar_promotion.json", "curves", "knnfar1", -1, "survives_pct")
    pct("finder", "Pixel finder (dev)", 128, P,
        "finder_promotion_s7.json", "rows", "finder_tv", "survives_pct")
    pct("finder_cf", "Centred-Frozen RSA, finder comparison (dev)", 128, P,
        "finder_promotion_s7.json", "rows", "arm 5", "survives_pct")

    cert = get("probe_certify.json")
    s = cert["sets"]["4,2/attn"]
    CORRECT["cert_clean"] = set(s["defended_clean_correct"])
    CORRECT["plain_clean"] = set(cert["undefended_clean_correct"])
    out += [("cert", "Provable defence, guaranteed correct (36 blocks)", len(s["certified"]), 512, P),
            ("cert_clean", "Provable defence, no attack", len(s["defended_clean_correct"]), 512, P),
            ("plain_clean", "Plain model, no attack", len(cert["undefended_clean_correct"]), 512, E)]

    cam = get("probe_camouflage.json")
    locs = list(cam["unconstrained"])
    out.append(("camo_none", "Camouflage (dev): broken, no cap",
                union(*(cam["unconstrained"][l]["wrong"] for l in locs)), 128, P))
    for sel in ("anomaly", "knnfar1", "tv"):
        for q in ("1.0", "0.9", "0.75", "0.5"):
            out.append((f"camo_{sel}_{q}", f"Camouflage (dev): broken, {sel} cap q={q}",
                        union(*(cam["capped"][f"{l}/{sel}/{q}"]["success"] for l in locs)), 128, P))
    return out


PAIRS = [("released_grad", "correct_grad"), ("oracle", "correct_grad"),
         ("cf_usual", "cf_all"), ("transfer_cf", "transfer_ref"), ("finder", "finder_cf"),
         ("cert_clean", "plain_clean")]
BARS = [("knnfar1", 55.0, "below")]


def self_check():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-4 and abs(hi - 0.5962) < 1e-4
    assert wilson(0, 512)[0] == 0 and abs(wilson(0, 512)[1] - 0.00745) < 1e-4
    assert abs(mcnemar_bound(5, 0, 10) - 0.0625) < 1e-12          # b=5, c=0
    assert mcnemar_bound(3, 3, 10) == 1.0
    assert abs(mcnemar_exact(10, 2) - 0.03857421875) < 1e-12


def main():
    self_check()
    figs = {k: (label, kk, n, st) for k, label, kk, n, st in figures()}
    rows = []
    print("| Figure | Count | % | 95% CI | Status |\n|---|---|---|---|---|")
    for key, (label, k, n, st) in figs.items():
        lo, hi = wilson(k, n)
        rows.append(dict(key=key, label=label, k=k, n=n, pct=100 * k / n,
                         ci=[100 * lo, 100 * hi], status=st))
        print(f"| {label} | {k}/{n} | {100 * k / n:.2f} | {100 * lo:.2f} to {100 * hi:.2f} | {st} |")

    tests = []
    print("\n| Paired comparison (same images) | Counts | McNemar p |\n|---|---|---|")
    for a, b in PAIRS:
        (la, ka, na, _), (lb, kb, nb, _) = figs[a], figs[b]
        assert na == nb
        extra = {}
        if a in CORRECT and b in CORRECT:
            extra = dict(only_a=len(CORRECT[a] - CORRECT[b]), only_b=len(CORRECT[b] - CORRECT[a]))
            p, kind = mcnemar_exact(extra["only_a"], extra["only_b"]), "exact"
        else:
            p, kind = mcnemar_bound(ka, kb, na), "at most"
        tests.append(dict(a=a, b=b, k=[ka, kb], n=na, mcnemar_p=p, p_kind=kind, **extra))
        print(f"| {la} vs {lb} | {ka} vs {kb} of {na} | {kind} {p:.1e} |")
    for key, bar, side in BARS:
        label, k, n, _ = figs[key]
        p = binom_cdf(k, n, bar / 100)                                    # one-sided, below
        tests.append(dict(a=key, bar_pct=bar, side=side, k=k, n=n, binomial_p=p))
        print(f"| {label} vs fixed bar {bar:g}% | {k} of {n} | {p:.1e} (one-sided) |")

    (R / "confidence_intervals.json").write_text(
        json.dumps(dict(purpose="95% Wilson intervals and paired tests for reported figures",
                        z=Z, figures=rows, tests=tests), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
