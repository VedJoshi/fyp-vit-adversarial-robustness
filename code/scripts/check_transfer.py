"""Check defense swaps, patch support, replay identity and failure unions."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace  # noqa: E402

from fyp import attacks, config, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402
from _stub_vit import _Recorder, _StubViT  # noqa: E402
import transfer_attack as T  # noqa: E402

SIZE, TOP, LEFT = 20, 60, 60

torch.manual_seed(0)
model = _StubViT().eval()
x = torch.rand(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
y = torch.tensor([1, 4])

target_cfg = replace(rsa.RSAConfig.for_patch(SIZE, window_rule="centred"),
                    freeze_mask_after=0)

# 1. The swap reaches the model
with rsa.enable(model, target_cfg) as handle:
    with torch.no_grad():
        model(x)
    frozen_masks = {i: m.clone() for i, m in handle.masks().items()}
    assert all(torch.equal(frozen_masks[0], m) for m in frozen_masks.values()), \
        "freeze_mask_after=0 did not make every block reuse block 0's mask"

    mean_cfg = replace(target_cfg, **T.SOURCES["mean"])
    handle.set_config(mean_cfg)
    with torch.no_grad():
        model(x)
    mean_masks = handle.masks()
    assert not all(torch.equal(mean_masks[0], m) for m in mean_masks.values()), \
        "a frozen mask survived set_config into the unfrozen source"

    handle.set_config(replace(target_cfg, **T.SOURCES["undefended"]))
    with torch.no_grad():
        model(x)
    assert all(m is None for m in handle.masks().values()), \
        "the undefended source still masked tokens"

    handle.set_config(target_cfg)
    with torch.no_grad():
        model(x)
    assert all(torch.equal(frozen_masks[i], m) for i, m in handle.masks().items()), \
        "swapping back to the target did not restore its masks"
print("1. set_config swap          OK  frozen/unfrozen/undefended differ, target restores")

# 2 and 3. The image is carried across unchanged, inside the patch support
with rsa.enable(model, target_cfg) as handle:
    source_cfg = replace(target_cfg, **T.SOURCES["mean"])
    config.seed_everything(paired_seed(SIZE, TOP, LEFT, 0))
    handle.set_config(source_cfg)
    adv = attacks.patch_autopgd(model, x, y, top=TOP, left=LEFT, size=SIZE, steps=3)
    before = adv.detach().clone()
    handle.set_config(target_cfg)
    with torch.no_grad():
        attacks.accuracy(model, adv.detach(), y)
    assert torch.equal(before, adv.detach()), "the evaluation pass modified the image"
print("2. image carried across     OK  bit-identical through the target's forward pass")

m = attacks.patch_mask(x.shape, TOP, LEFT, SIZE, x.device).bool().expand_as(x)
for name in ("undefended", "mean", "centred"):
    with rsa.enable(model, target_cfg) as handle:
        config.seed_everything(paired_seed(SIZE, TOP, LEFT, 0))
        handle.set_config(replace(target_cfg, **T.SOURCES[name]))
        a = attacks.patch_autopgd(model, x, y, top=TOP, left=LEFT, size=SIZE, steps=3)
    assert torch.equal(a[~m.expand_as(a)], x[~m.expand_as(x)]), \
        f"source {name!r} changed pixels outside the {SIZE}px patch"
    assert not torch.equal(a[m.expand_as(a)], x[m.expand_as(x)]), \
        f"source {name!r} changed nothing inside the patch"
print(f"3. patch support            OK  3 sources touch only the {SIZE}x{SIZE} square "
      f"at ({TOP},{LEFT})")

# 4. Shared initial patch
inits = {}
for name in ("mean", "self"):
    rec = _Recorder(model)
    with rsa.enable(model, target_cfg) as handle:
        handle.set_config(replace(target_cfg, **T.SOURCES[name]))
        config.seed_everything(paired_seed(SIZE, TOP, LEFT, 0))
        attacks.patch_autopgd(rec, x, y, top=TOP, left=LEFT, size=SIZE, steps=1)
    inits[name] = rec.first
assert torch.equal(inits["mean"], inits["self"]), \
    "the transfer and the target's own attack started from different patches"
assert paired_seed(SIZE, TOP, LEFT, 0) != paired_seed(SIZE, TOP, LEFT, 32), \
    "paired_seed collides across batch offsets"
print("4. shared initial patch     OK  transfer and self-attack start identically")

# 5. Union arithmetic, the clean-error symmetry, and the guards
locs = [(0, 0), (0, 20)]
CLEAN = {9}
per_loc = {(0, 0): {1, 2} | CLEAN, (0, 20): {2, 3} | CLEAN}     # transfer union {1,2,3,9}
tgt_raw = {(0, 0): {2}, (0, 20): {2, 4}}                        # record, clean not folded in
tgt = T.with_clean(tgt_raw, CLEAN)                              # target union {2,4,9}
assert tgt[(0, 0)] == {2, 9} and tgt[(0, 20)] == {2, 4, 9}, "with_clean did not fold"

s5 = T.summarise(100, locs, {"mean": per_loc}, tgt, set())
d = s5["by_source"]["mean"]
assert d["alone"]["worst_case_pct"] == 96.0, d["alone"]          # {1,2,3,9}
assert s5["target_own_attack"]["worst_case_pct"] == 97.0, s5["target_own_attack"]
assert s5["suite_union"]["worst_case_pct"] == 95.0, s5["suite_union"]   # {1,2,3,4,9}
assert d["images_broken_only_by_this_source"] == [1, 3]
assert s5["images_broken_only_by_target_attack"] == [4]
assert s5["target_own_attack"]["mean_per_location_pct"] == 97.5, s5["target_own_attack"]
assert d["alone"]["mean_per_location_pct"] == 97.0, d["alone"]
print("5. union and clean symmetry OK  suite 95.0% beats source 96.0% and target 97.0%; "
      "both means fold clean errors")

# 6. A perfect self replay must report nothing
replay = T.summarise(100, locs, {"self": dict(tgt)}, tgt, set())
r = replay["by_source"]["self"]
assert r["locations_where_this_source_found_more"] == [],     "a perfect replay was reported as finding extra failures"
assert r["images_broken_only_by_this_source"] == []
assert replay["images_broken_only_by_target_attack"] == []
assert r["alone"] == replay["target_own_attack"] == replay["suite_union"],     "a perfect replay disagreed with the target on worst case or mean"
assert T.check_self_replay(locs, dict(tgt), tgt) == [], "a perfect replay was flagged"
skew = {(0, 0): {2, 9}, (0, 20): {2, 4, 9}}
skew[(0, 0)] = {4, 9}                     # swaps which image fails here; same sizes
bad = T.check_self_replay(locs, skew, tgt)
assert len(bad) == 1 and bad[0]["only_in_replay"] == [4] and bad[0]["only_in_record"] == [2], bad
same = T.summarise(100, locs, {"self": skew}, tgt, set())
assert (same["by_source"]["self"]["alone"]["mean_per_location_pct"]
        == replay["target_own_attack"]["mean_per_location_pct"]),     "the skewed replay was meant to share the mean, so the identity test is what catches it"
print("6. self-replay acceptance   OK  perfect replay reports 0 extra locations; a "
      "one-image swap with an identical mean is still caught")

# 7. The guards
base = {"n_images": 512, "sample_seed": 0, "wnids_sha1": "abc",
        "class_draw": "local_seed1234", "loc_stride": 1}
T.check_comparable(base, dict(base))
for field, other in (("n_images", 128), ("sample_seed", 1), ("wnids_sha1", "def"),
                     ("class_draw", "rsa_seed0")):
    try:
        T.check_comparable(base, dict(base, **{field: other}))
    except SystemExit:
        pass
    else:
        raise AssertionError(f"a differing {field} was allowed into a union")
T.check_comparable(base, dict(base, loc_stride=3))

rec = {"window": 3, "window_rule": "centred", "renormalise": "uniform",
       "score_frame": "global", "backward": "intended", "freeze_mask_after": 0,
       "checkpoint": None}
T.check_target_config(target_cfg, rec)
for field, other in (("window_rule", "ceil+1"), ("freeze_mask_after", 5),
                     ("backward", "released"), ("renormalise", "zero"),
                     ("score_frame", "per-head"), ("window", 5),
                     ("checkpoint", "rsa.pyth")):
    try:
        T.check_target_config(target_cfg, dict(rec, **{field: other}))
    except SystemExit:
        pass
    else:
        raise AssertionError(f"an evaluation target differing in {field} was accepted")

prot = dict(base, attack="patch_autopgd(dlr)", steps=100, init="uniform",
            schedule="counter", batch_sizes=[32] * 16, attack_survivors_only=False,
            init_rng="paired_seed(seed, size, top, left, batch offset)")
T.check_self_protocol(prot, prot, {1, 2}, {1, 2})
for field, other in (("steps", 50), ("batch_sizes", [64] * 8), ("schedule", "none"),
                     ("attack", "patch_pgd"), ("attack_survivors_only", True)):
    try:
        T.check_self_protocol(dict(prot, **{field: other}), prot, {1, 2}, {1, 2})
    except SystemExit:
        pass
    else:
        raise AssertionError(f"a self control differing in {field} was accepted")
try:
    T.check_self_protocol(prot, prot, {1, 3}, {1, 2})
except SystemExit:
    pass
else:
    raise AssertionError("a self control with different clean-error identities was accepted")
print("7. guards                   OK  4 population, 7 target-defense, 6 self-protocol "
      "mismatches each refuse; a coarser grid is allowed")


# 8. Reject mismatched targets and report missing settings in older records.
stored = {"target_tag": "gate_a_diag_centred_frz0", "target_window": 3,
          "target_window_rule": "centred", "target_renormalise": "uniform",
          "target_score_frame": "global", "target_backward": "intended",
          "target_freeze_mask_after": 0}
assert T.check_recorded_target("t", stored, rec, "gate_a_diag_centred_frz0") == []
for field, other in (("target_window_rule", "ceil+1"), ("target_freeze_mask_after", None),
                     ("target_backward", "released"), ("target_renormalise", "zero"),
                     ("target_score_frame", "per-head"), ("target_window", 5)):
    try:
        T.check_recorded_target("t", dict(stored, **{field: other}), rec,
                                "gate_a_diag_centred_frz0")
    except SystemExit:
        pass
    else:
        raise AssertionError(f"a stored run evaluated under {field}={other!r} was combined")
try:
    T.check_recorded_target("t", dict(stored, target_tag="gate_a_geom20"), rec,
                            "gate_a_diag_centred_frz0")
except SystemExit:
    pass
else:
    raise AssertionError("a run recorded against another target was combined")
old_record = {"target_tag": "gate_a_diag_centred_frz0", "target_window_rule": "centred",
              "target_freeze_mask_after": 0}
assert sorted(T.check_recorded_target("t", old_record, rec, "gate_a_diag_centred_frz0")) \
    == ["backward", "renormalise", "score_frame", "window"]
print("8. combine target guard     OK  6 defense mismatches and a wrong target tag refuse; "
      "an older record names its 4 unverified settings")

# 9. The black-box row holds transfers only, and mixed records are refused
mixed = T.summarise(100, locs, {"mean": per_loc, "self": dict(tgt)}, tgt, set(),
                    transfers={"mean"})
assert mixed["transfer_sources"] == ["mean"], mixed["transfer_sources"]
assert mixed["transfer_only_union"] == s5["by_source"]["mean"]["alone"], "self leaked in"
none = T.summarise(100, locs, {"self": dict(tgt)}, tgt, set(), transfers=set())
assert "transfer_only_union" not in none, "a self replay was labelled black-box"
ref_fp = {"window": 3, "window_rule": "ceil+1", "freeze_mask_after": None}
arm5_fp = {"window": 3, "window_rule": "centred", "freeze_mask_after": 0}
mean_run = {"transferred": True, "source_window": 3, "source_window_rule": "ceil+1",
            "source_freeze_mask_after": None}
undef_run = {"transferred": True, "source_window": 0, "source_window_rule": "ceil+1",
             "source_freeze_mask_after": None}
assert T.is_transfer(mean_run, arm5_fp) and not T.is_transfer(mean_run, ref_fp), \
    "an alias of the target was counted as a transfer"
assert T.is_transfer(undef_run, ref_fp)
assert not T.is_transfer(dict(mean_run, transferred=False), arm5_fp)
import json, tempfile                                                    # noqa: E402
with tempfile.TemporaryDirectory() as d:
    hdr = {"record": "protocol", "size": 20, "clean_wrong": [9], "fingerprint": dict(base)}
    loc = {"record": "location", "top": 0, "left": 0, "wrong": [1]}
    p = Path(d) / "x.progress.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (hdr, loc, hdr, loc)) + "\n")
    assert T.read_target("x", Path(d))[2]["size"] == 20, "an identical repeated header was refused"
    other = dict(hdr, fingerprint=dict(base, sample_seed=1))
    p.write_text("\n".join(json.dumps(r) for r in (hdr, loc, other, loc)) + "\n")
    try:
        T.read_target("x", Path(d))
    except SystemExit:
        pass
    else:
        raise AssertionError("a file holding two experiments was read as one")
print("9. black-box row and record OK  self and target aliases stay out of the transfer "
      "union; a two-experiment file refuses")

# Exercise combine itself: filtering alone must not bypass a failed self control.
from contextlib import redirect_stdout                                  # noqa: E402
from io import StringIO                                                 # noqa: E402
from types import SimpleNamespace                                       # noqa: E402
from unittest.mock import patch                                         # noqa: E402

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    hdr = {"record": "protocol", "size": 20, "clean_wrong": [9],
           "fingerprint": dict(base, **rec)}
    loc = {"record": "location", "size": 20, "top": 0, "left": 0, "wrong": [1]}
    second = dict(loc, left=20, wrong=[2])

    def write_record(name, header, rows):
        (root / f"{name}.progress.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [header, *rows]) + "\n")

    write_record("target", hdr, [loc, second])
    self_hdr = dict(hdr, fingerprint=dict(base, source="self", transferred=False))
    transfer_hdr = dict(hdr, fingerprint=dict(base, source="undefended", **undef_run))
    write_record("self", self_hdr, [loc, dict(second, wrong=[3])])
    write_record("transfer", transfer_hdr, [loc])
    write_record("disjoint", dict(transfer_hdr, fingerprint=dict(
        transfer_hdr["fingerprint"], source="centred")), [second])
    with patch.object(T.config, "RESULTS_ROOT", root), redirect_stdout(StringIO()):
        for tags, message in [("self,transfer", "1"),
                              ("transfer,transfer", "two runs record source"),
                              ("transfer,disjoint", "no shared locations")]:
            try:
                T.combine(SimpleNamespace(target_tag="target", combine=tags, out=None))
            except SystemExit as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"combine accepted {tags}")
        with patch.object(T.results, "save") as save:
            T.combine(SimpleNamespace(target_tag="target", combine="transfer", out="out"))
            saved = save.call_args.args[1]
            assert saved["partial_locations"] and saved["locations"] == [[0, 0]]
            assert not saved["locations_reduced_to_intersection"]
    for rows in [[loc, dict(loc, wrong=[3])], [dict(loc, size=30)]]:
        write_record("bad", hdr, rows)
        try:
            T.read_target("bad", root)
        except SystemExit:
            pass
        else:
            raise AssertionError("conflicting locations or patch sizes were accepted")
    try:
        T.check_comparable(dict(base, size=20), dict(base, size=30))
    except SystemExit:
        pass
    else:
        raise AssertionError("different patch sizes were combined")
print("10. combine integration     OK  mixed self failure, duplicates, disjoint grids, "
      "partial-grid metadata and conflicting records")

print("\nall transfer checks passed")
