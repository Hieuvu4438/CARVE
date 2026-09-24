"""Generate T1–T8 Markdown tables from experiment/results/*.json."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXPERIMENT_ROOT, output_path

PENDING = "pending"
RESULTS = EXPERIMENT_ROOT / "results"


def read(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def md(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + (
        "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows) if rows else
        "| " + " | ".join([PENDING] + ["—"] * (len(headers) - 1)) + " |"
    ) + "\n"


def metric(result, key, field, *, spread=True):
    if result is None:
        return PENDING
    v = result["summary"][key][field]
    return f"{v['mean']:.2f} ± {v['std']:.2f}" if spread and result["n_runs"] > 1 else f"{v['mean']:.2f}"


def rule(result):
    if result is None or result.get("rule") is None:
        return "archive" if result and result.get("archive") else PENDING
    return Path(result["rule"]["path"]).name


def baseline_rows(archive, key, current):
    rows = []
    if archive:
        source = archive[key]
        for name, vals in source.items():
            if name == "CLAVE":
                continue
            if name == "CARVE-simple, seed 42":
                continue
            result = current.get(name)
            if result:
                keys = (["trigger_rougeL"] if key == "trigger_rougeL" else
                        ["arg_i_iou", "arg_c_iou"])
                label = name + (" (archive)" if result.get("archive") else "")
                rows.append([label, *(metric(result, k, f) for k in keys
                                     for f in ("p", "r", "f1")), rule(result)])
            else:
                rows.append([name + " (archive)", *vals, "archive"])
    return rows


def main_tables():
    a, main = read("published_baselines"), read("main")
    simple, orig, s42 = read("carve_simple"), read("carve_orig"), read("clave_s42")
    current = {"original CARVE (3 seeds)": orig, "CARVE-simple (3 seeds)": simple}
    t1 = baseline_rows(a, "trigger_rougeL", current)
    for label, result in (("CLAVE, 1 seed (archive)", s42), ("CLAVE, 3 seeds", main)):
        t1.append([label, *(metric(result, "trigger_rougeL", x) for x in ("p", "r", "f1")), rule(result)])
    t2 = baseline_rows(a, "argument_iou", current)
    for label, result in (("CLAVE, 1 seed (archive)", s42), ("CLAVE, 3 seeds", main)):
        t2.append([label, *(metric(result, k, f) for k in ("arg_i_iou", "arg_c_iou")
                            for f in ("p", "r", "f1")), rule(result)])
    t3 = []
    for mode in ("exact", "overlap", "scirex", "iou"):
        for label, result in (("CARVE-simple", simple), ("CLAVE", main)):
            t3.append([mode, label, *(metric(result, f"arg_{kind}_{mode}", field)
                                         for kind in ("i", "c") for field in ("p", "r", "f1")),
                       rule(result)])
    boot = read("bootstrap_main")
    t4 = []
    if boot:
        for comparison, metrics in boot["comparisons"].items():
            for key, value in metrics.items():
                t4.append([comparison, key, f"{value['delta']:+.2f}",
                           f"[{value['ci_low']:+.2f}, {value['ci_high']:+.2f}]",
                           f"{value['p_gt_zero']:.3f}",
                           f"[{value['sys_ci_low']:.2f}, {value['sys_ci_high']:.2f}]",
                           f"{value['oneie']:.2f}"])
    return {
        "T1": "# T1. Trigger ROUGE-L (test)\n\n" +
              md(["System", "P", "R", "F1", "Rule"], t1),
        "T2": "# T2. Arguments, IoU > 0.5 (test)\n\n" +
              md(["System", "Arg-I P", "Arg-I R", "Arg-I F1", "Arg-C P", "Arg-C R", "Arg-C F1", "Rule"], t2),
        "T3": "# T3. Matching modes (test)\n\n" +
              md(["Mode", "System", "Arg-I P", "Arg-I R", "Arg-I F1",
                  "Arg-C P", "Arg-C R", "Arg-C F1", "Rule"], t3),
        "T4": "# T4. Paired window bootstrap, 5,000 samples\n\n" +
              md(["Comparison", "Metric", "Δ", "95% CI", "P(Δ > 0)",
                  "CLAVE 95% CI", "OneIE"], t4),
    }


def ablation_table():
    names = [
        ("CLAVE full", "main"),
        ("− verifier (CARVE-simple)", "carve_simple"),
        ("− cross-fitting", "a1"),
        ("− proposer features", "a2"),
        ("α = 1", "a3"),
        ("α = 0", "a4"),
        ("λ = 0", "a5"),
        ("α = 1, λ = 0", "a6"),
        ("NMS iou", "a7_iou"),
        ("NMS wis", "a7_wis"),
    ]
    full = read("main")
    main_bootstrap = read("bootstrap_main")
    rows = []
    for label, name in names:
        result = read(name)
        content = result["rule"]["content"] if result and result.get("rule") else {}
        dev = content.get("dev_arg_c_iou_mean", content.get("dev_mean"))
        dev_std = content.get("dev_arg_c_iou_std", content.get("dev_std"))
        dev_str = (f"{dev:.2f} ± {dev_std:.2f}" if dev is not None and dev_std is not None
                   else PENDING)
        boot = read(f"bootstrap_{name}")
        ci = PENDING
        if name == "main":
            ci = "+0.00 [+0.00, +0.00]"
        elif name == "carve_simple" and main_bootstrap:
            x = main_bootstrap["comparisons"]["CLAVE vs CARVE-simple"]["arg_c_iou"]
            ci = f"{-x['delta']:+.2f} [{-x['ci_high']:+.2f}, {-x['ci_low']:+.2f}]"
        elif boot and "arg_c_iou" in boot:
            x = boot["arg_c_iou"]
            ci = f"{x['delta']:+.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]"
        elif result and full:
            ci = f"{metric(result, 'arg_c_iou', 'f1', spread=False)} vs full; CI pending"
        rows.append([label, dev_str, metric(result, "arg_i_iou", "f1"),
                     *(metric(result, "arg_c_iou", x) for x in ("p", "r", "f1")), ci,
                     rule(result)])
    return "# T5. Main ablations\n\n" + md(
        ["Variant", "Dev Arg-C F1", "Test Arg-I F1", "Test Arg-C P",
         "Test Arg-C R", "Test Arg-C F1", "Δ / 95% CI", "Rule"], rows)


def proposer_table():
    archive = read("published_baselines")
    rows = []
    for label, name in (("argmax, τ=0, min_len=1", "proposer_argmax"),
                        ("calibrated CARVE-simple", "carve_simple")):
        result = read(name)
        content = result["rule"]["content"] if result and result.get("rule") else {}
        dev = content.get("dev_arg_c_iou_mean", content.get("dev_mean"))
        dev_sd = content.get("dev_arg_c_iou_std", content.get("dev_std"))
        dev_text = (f"{dev:.2f} ± {dev_sd:.2f}" if dev is not None and dev_sd is not None
                    else PENDING)
        rows.append([label, dev_text, metric(result, "arg_i_iou", "f1"),
                     metric(result, "arg_c_iou", "f1"), rule(result)])
    out = "# T6. Proposer design\n\n" + md(
        ["Decode", "Dev Arg-C F1", "Test Arg-I F1", "Test Arg-C F1", "Rule"], rows)
    p2 = [("Two heads + event conditioning (archive)", read("carve_orig")),
          ("Two heads, no event conditioning (T-P2a)", read("abl_no_type_cond")),
          ("One head + event conditioning (T-P2b)", read("abl_single_head")),
          ("One head, no event conditioning (archive)", read("carve_simple"))]
    out += "\n2×2 test comparison (T-P2 fills the two previously missing cells; each row has one shared dev-frozen rule):\n\n"
    design_rows = []
    for label, result in p2:
        content = result["rule"]["content"] if result and result.get("rule") else {}
        dev = content.get("dev_arg_c_iou_f1_mean", content.get("dev_mean",
              content.get("dev_arg_c_iou_mean")))
        dev_sd = content.get("dev_arg_c_iou_f1_std", content.get("dev_std",
                 content.get("dev_arg_c_iou_std")))
        dev_text = (f"{dev:.2f} ± {dev_sd:.2f}" if dev is not None and dev_sd is not None
                    else PENDING)
        design_rows.append([label, dev_text, metric(result, "arg_i_iou", "f1"),
                            metric(result, "arg_c_iou", "f1"), rule(result)])
    out += md(["Design", "Dev Arg-C F1", "Test Arg-I F1", "Test Arg-C F1", "Rule"],
              design_rows)
    out += ("\nThe full and CARVE-simple rows use archived checkpoints; T-P2a/b retrain "
            "the archived code in an isolated experiment copy. GPU training is "
            "not bit deterministic, so this is a descriptive factorial comparison.\n")
    design = archive["proposer_design_same_grid"] if archive else {}
    out += "\nSame-grid 2×2 development comparison (archive):\n\n"
    out += md(["Design", "Seed 42", "Seed 13", "Seed 101", "Dev mean ± std",
               "Δ vs full"], [[name, *vals] for name, vals in design.items()])
    factors = archive["proposer_single_factor_dev"] if archive else {}
    out += "\nSingle-factor development ablations (archive; no new test):\n\n"
    out += md(["Configuration", "Dev Arg-C F1", "Δ vs full"],
              [[name, *vals] for name, vals in factors.items()])
    out += ("\nThe archive's 2×2 table reports 48.31 ± 1.62 for the single-head-only "
            "cell; its earlier Table 12 reports 48.34 ± 1.60. The plan quotes "
            "the latter. Both source values are preserved.\n")
    return out


def analysis_table():
    result = read("analysis")
    if result is None:
        return "# T7. Error analysis\n\n" + PENDING + "\n"
    runs = result["per_run"]

    def aggregate(key, branch=None):
        values = [r[key][branch] if branch else r[key] for r in runs]
        categories = sorted(set().union(*(v.keys() for v in values)))
        rows = []
        for category in categories:
            row = [category]
            for field in ("p", "r", "f1"):
                nums = [v[category][field] for v in values if category in v]
                row.append(f"{sum(nums) / len(nums):.2f}")
            rows.append(row)
        return rows

    out = "# T7. Error analysis (test; descriptive only)\n\n"
    out += "Rolewise Arg-C IoU:\n\n" + md(["Role", "P", "R", "F1"], aggregate("rolewise"))
    for group, label in (("eventtype", "Event type"), ("domain", "Domain")):
        for branch in ("Arg_I", "Arg_C"):
            out += f"\n{label}, {branch} IoU:\n\n" + md(
                [label, "P", "R", "F1"], aggregate(group, branch))
    lengths = [name for name in ("1–4", "5–9", "10–19", "20–39", "≥40")
               if name in runs[0]["length_recall"]]
    out += "\nGold span length recall:\n\n" + md(
        ["Length (tokens)", "Gold", "Recall %"],
        [[name, runs[0]["length_recall"][name]["gold"],
          f"{sum(r['length_recall'][name]['recall'] for r in runs) / len(runs):.2f}"]
         for name in lengths])
    errors = [name for name in ("correct", "boundary", "wrong_role", "extra", "missing")
              if any(name in r["errors"] for r in runs)]
    out += "\nError buckets (mean count per run):\n\n" + md(
        ["Category", "Count"],
        [[name, f"{sum(r['errors'].get(name, 0) for r in runs) / len(runs):.2f}"]
         for name in errors])
    out += ("\nError buckets and span-length recall use a local one-to-one "
            "diagnostic assignment; the P/R/F1 sections above use the official scorer.\n")
    shift = result.get("precision_recall_shift_vs_baseline") or {}
    out += "\nP/R shift from CARVE-simple (mean percentage points):\n\n" + md(
        ["Metric", "ΔP", "ΔR", "ΔF1"],
        [[name, *(f"{v[field]:+.2f}" for field in ("p", "r", "f1"))]
         for name, v in shift.items()])
    oracle = read("oracle_dev")
    out += "\nDev candidate oracles:\n\n" + md(
        ["Seed", "Frozen rule Arg-C F1", "Oracle keep + role Arg-C F1",
         "Oracle type only Arg-C F1", "Oracle all three Arg-C F1", "Rule"],
        [[seed, f"{x['baseline_arg_c_iou']['f1']:.2f}",
          f"{x['keep_role_arg_c_iou']['f1']:.2f}",
          f"{x['type_oracle_arg_c_iou']['f1']:.2f}",
          f"{x['joint_oracle_arg_c_iou']['f1']:.2f}", oracle["rule"]]
         for seed, x in oracle["seeds"].items()] if oracle else [])
    return out


def candidate_table():
    result = read("candidate_stats")
    if result is None:
        return "# T8. Candidate statistics and compute cost\n\n" + PENDING + "\n"
    rows = []
    for seed, parts in result["seeds"].items():
        for name, value in parts.items():
            rows.append([seed, name, value.get("windows", PENDING),
                         value.get("candidates", PENDING), value.get("per_window", PENDING),
                         value.get("positive_pct", PENDING)])
    out = "# T8. Candidate statistics and compute cost\n\n" + md(
        ["Seed", "Set", "Windows", "Candidates", "Per window", "Positive %"], rows)
    costs = result.get("gpu_seconds_from_logs", {})
    cost_rows = [[seed, stage, f"{seconds:.0f}" if seconds is not None else PENDING]
                 for seed, stages in costs.items() for stage, seconds in stages.items()]
    out += "\nElapsed training time from GPU-run logs (wall-clock seconds):\n\n"
    out += md(["Seed", "Stage", "Seconds"], cost_rows)
    out += "\nParameters (seed-42 checkpoint architecture):\n\n"
    out += md(["Model", "Parameters"],
              [[name, f"{count:,}"] for name, count in result.get("parameters", {}).items()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-complete", action="store_true")
    a = ap.parse_args()
    tables = main_tables()
    tables.update({"T5": ablation_table(), "T6": proposer_table(),
                   "T7": analysis_table(), "T8": candidate_table()})
    if a.require_complete and any(PENDING in content for content in tables.values()):
        raise RuntimeError("results are incomplete; refusing to claim complete tables")
    for name, content in tables.items():
        out = output_path("tables", f"{name}.md")
        out.write_text(content)
        print(out)


if __name__ == "__main__":
    main()
