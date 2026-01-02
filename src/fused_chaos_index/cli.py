from __future__ import annotations

import argparse
import json
from pathlib import Path

from .operational import StreamlinedFCIPipeline
from .syk_collatz import compute_syk_collatz_fci_constant
from .meta_tools import ablation_sensitivity, combine_evidence_heuristic, keyword_risk_audit
from .gate import check_frontier_manifest, check_universality_manifest
from .tier1.extract_radec import extract_radec_to_npz
from .tier1.add_quantum_mass import add_quantum_mass_to_catalog_npz
from .tier1.score_frontier import load_thresholds, score_frontier_manifest
from .tier1.score_single import score_single_artifact
from .suite import default_run_dir, run_public_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fci", description="Public Fused Chaos Index utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    op = sub.add_parser("operational", help="Run the streamlined operational FCI demo")
    op.add_argument("--n-galaxies", type=int, default=2000)
    op.add_argument("--k", type=int, default=10, help="k-NN neighbors")
    op.add_argument("--seed", type=int, default=42)

    sc = sub.add_parser("syk-collatz", help="Compute SYK–Collatz constant-approximation FCI")
    sc.add_argument("--N", type=int, default=32)
    sc.add_argument("--kurt-collatz", type=float, default=3.72)
    sc.add_argument("--kurt-syk", type=float, default=-1.23)
    sc.add_argument("--method", choices=["analytic", "quad", "mc"], default="analytic")

    meta = sub.add_parser("meta", help="Meta-tools (audits, evidence aggregation, robustness diagnostics)")
    meta_sub = meta.add_subparsers(dest="meta_cmd", required=True)

    audit = meta_sub.add_parser("audit", help="Keyword risk audit for a string")
    audit.add_argument("text")

    comb = meta_sub.add_parser("combine", help="Combine evidence strengths (heuristic)")
    comb.add_argument("--prior", type=float, default=0.5)
    comb.add_argument("evidences", nargs="+", type=float)

    abl = meta_sub.add_parser("ablation", help="Ablation sensitivity diagnostic")
    abl.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    abl.add_argument("--n-trials", type=int, default=200)
    abl.add_argument("--seed", type=int, default=42)
    abl.add_argument("x", nargs="+", type=float)
    abl.add_argument("--y", nargs="+", type=float, required=True)

    tier1 = sub.add_parser("tier1", help="Tier-1 helpers (catalog conversion, scoring)")
    tier1_sub = tier1.add_subparsers(dest="tier1_cmd", required=True)

    er = tier1_sub.add_parser("extract-radec", help="Extract RA/Dec from a table catalog into NPZ")
    er.add_argument("--catalog", type=Path, required=True)
    er.add_argument("--out-npz", type=Path, required=True)
    er.add_argument("--ra-col", type=str, default=None)
    er.add_argument("--dec-col", type=str, default=None)
    er.add_argument("--max-rows", type=int, default=0)

    ss = tier1_sub.add_parser("score-single", help="Score a single artifact JSON vs a Tier-1 prediction card")
    ss.add_argument("--prediction-card-json", type=Path, required=True)
    ss.add_argument("--artifact-json", type=Path, required=True)
    ss.add_argument("--out-json", type=Path, default=Path("tier1_single_kappa_score.json"))

    sf = tier1_sub.add_parser("score-frontier", help="Score a Frontier manifest JSON vs a Tier-1 prediction card")
    sf.add_argument("--prediction-card", type=Path, required=True)
    sf.add_argument("--frontier-manifest", type=Path, required=True)
    sf.add_argument("--out-json", type=Path, default=Path("tier1_frontier_accuracy.json"))

    aqm = tier1_sub.add_parser("add-quantum-mass", help="Compute quantum_mass from a local NPZ catalog (positions or RA/Dec)")
    aqm.add_argument("--input-npz", type=Path, required=True)
    aqm.add_argument("--out-npz", type=Path, required=True)
    aqm.add_argument("--k", type=int, default=10)
    aqm.add_argument("--n-modes", type=int, default=10)
    aqm.add_argument("--threshold", type=float, default=5e-7, help="Only used to compute dark_percent")
    aqm.add_argument("--force", action="store_true", help="Overwrite existing quantum_mass/eigenvalues if present")

    tier2 = sub.add_parser("tier2", help="Tier-2 analyses (artifact-driven, offline-first)")
    tier2_sub = tier2.add_subparsers(dest="tier2_cmd", required=True)

    t2_u = tier2_sub.add_parser(
        "universality-sweep",
        help="Tier-2 Path 1: compare multiple quantum-mass artifacts under a shared threshold",
    )
    t2_u.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    t2_u.add_argument("--inputs", type=Path, nargs="+", required=True, help="Input NPZ files (must contain quantum_mass/mass/M)")
    t2_u.add_argument("--threshold", type=float, default=5e-7)
    t2_u.add_argument("--plot", action="store_true", help="Write a small PNG plot (requires matplotlib)")

    t2_s = tier2_sub.add_parser(
        "stopping-time",
        help="Tier-2 Path 3: stopping time vs quantum mass association test (artifact-driven)",
    )
    t2_s.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    t2_s.add_argument("--input", type=Path, required=True, help="Input NPZ (must contain quantum_mass/mass/M)")
    t2_s.add_argument("--sample-size", type=int, default=200_000)
    t2_s.add_argument("--max-steps", type=int, default=5000)
    t2_s.add_argument("--seed", type=int, default=42)
    t2_s.add_argument("--bootstrap", action="store_true")
    t2_s.add_argument("--bootstrap-iters", type=int, default=200)

    t2_f = tier2_sub.add_parser(
        "fingerprint",
        help="Tier-2 Path 2: compare Collatz vs cosmic artifacts (distribution fingerprints + correlations)",
    )
    t2_f.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    t2_f.add_argument(
        "--collatz",
        type=Path,
        required=True,
        help="Collatz NPZ containing quantum_mass and optionally eigenvalues",
    )
    t2_f.add_argument(
        "--smacs",
        type=Path,
        required=True,
        help="SMACS NPZ containing quantum_mass and kappa",
    )
    t2_f.add_argument(
        "--flamingo",
        type=Path,
        default=None,
        help="Optional FLAMINGO NPZ artifact (loaded if present)",
    )
    t2_f.add_argument("--threshold", type=float, default=5e-7)
    t2_f.add_argument("--k", type=int, default=10, help="Only used if SMACS quantum_mass must be computed")
    t2_f.add_argument("--n-modes", type=int, default=10, help="Only used if SMACS quantum_mass must be computed")

    t2_c = tier2_sub.add_parser(
        "collatz-summary",
            help="Summarize a Collatz NPZ (dark-matter stats, and spectral gap ratio if available)",
    )
    t2_c.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    t2_c.add_argument(
        "--collatz",
        type=Path,
        required=True,
        help="Collatz NPZ containing quantum_mass (or mass/M), and optionally eigenvalues",
    )
    t2_c.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline NPZ for comparison (e.g., 10M)",
    )
    t2_c.add_argument("--threshold", type=float, default=5e-7)
    t2_c.add_argument("--cosmic-target", type=float, default=68.0)
    t2_c.add_argument(
        "--runtime-seconds",
        type=float,
        default=None,
        help="Optional runtime (seconds) to record in the manifest (not inferred from NPZ)",
    )

    t2_p = tier2_sub.add_parser(
        "plot-16m",
        help="Write a lightweight discovery plot for a Collatz NPZ (requires matplotlib; skip-safe)",
    )
    t2_p.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    t2_p.add_argument(
        "--collatz",
        type=Path,
        required=True,
        help="Collatz NPZ containing quantum_mass (or mass/M), optionally eigenvalues",
    )
    t2_p.add_argument("--threshold", type=float, default=5e-7)
    t2_p.add_argument("--cosmic-target", type=float, default=68.0)

    gate = sub.add_parser("gate", help="Falsification/null-test presence gate for suite manifests")
    gate.add_argument("--frontier-manifest", type=Path, default=Path("validation_results/frontier_evidence_suite_manifest.json"))
    gate.add_argument("--universality-manifest", type=Path, default=Path("validation_results/universality_ground_truth_suite_manifest.json"))

    suite = sub.add_parser("suite", help="Offline-first suite runner (writes a single manifest)")
    suite_sub = suite.add_subparsers(dest="suite_cmd", required=True)

    suite_run = suite_sub.add_parser("run", help="Run the public suite")
    suite_run.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    suite_run.add_argument(
        "--profile",
        choices=["smoke", "offline", "full"],
        default="smoke",
        help="Which suite profile to run",
    )
    suite_run.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network downloads for validators (default: blocked)",
    )
    suite_run.add_argument(
        "--frontier-clusters-json",
        type=Path,
        default=None,
        help="Clusters spec JSON for Frontier evidence step (only used in --profile full)",
    )
    suite_run.add_argument(
        "--tng-base-path",
        type=Path,
        default=Path("./TNG300-1/output"),
        help="Path to local TNG dataset root OR a derived NPZ artifact with positions+mass",
    )
    suite_run.add_argument("--skip-tng", action="store_true", help="Skip running the TNG validator")
    suite_run.add_argument("--n-galaxies", type=int, default=2000)
    suite_run.add_argument("--k", type=int, default=10)
    suite_run.add_argument("--seed", type=int, default=42)
    suite_run.add_argument("--n-eigenstates", type=int, default=40)
    suite_run.add_argument("--syk-N", type=int, default=32)
    suite_run.add_argument("--kurt-collatz", type=float, default=3.72)
    suite_run.add_argument("--kurt-syk", type=float, default=-1.23)
    suite_run.add_argument("--method", choices=["analytic", "quad", "mc"], default="analytic")
    suite_run.add_argument(
        "--frontier-manifest",
        type=Path,
        default=Path("validation_results/frontier_evidence_suite_manifest.json"),
    )
    suite_run.add_argument(
        "--universality-manifest",
        type=Path,
        default=Path("validation_results/universality_ground_truth_suite_manifest.json"),
    )

    suite_uni = suite_sub.add_parser("universality", help="Run universality ground-truth suite")
    suite_uni.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    suite_uni.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network downloads for validators (default: blocked)",
    )
    suite_uni.add_argument(
        "--tng-base-path",
        type=Path,
        default=Path("./TNG300-1/output"),
        help="Path to local TNG dataset root",
    )
    suite_uni.add_argument("--skip-tng", action="store_true", help="Skip running the TNG validator")
    suite_uni.add_argument("--k", type=int, default=10)
    suite_uni.add_argument("--n-modes", type=int, default=10)
    suite_uni.add_argument("--bolshoi-max-n", type=int, default=5000)

    suite_front = suite_sub.add_parser("frontier", help="Run Frontier evidence suite on local NPZ artifacts")
    suite_front.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    suite_front.add_argument(
        "--clusters-json",
        type=Path,
        required=True,
        help="JSON describing clusters and local NPZ paths",
    )
    suite_front.add_argument("--k", type=int, default=10)
    suite_front.add_argument("--n-modes", type=int, default=10)
    suite_front.add_argument("--n-perm", type=int, default=2000)
    suite_front.add_argument("--seed", type=int, default=42)

    validate = sub.add_parser("validate", help="Run validators (safe: SKIP if deps/data missing)")
    validate_sub = validate.add_subparsers(dest="validate_cmd", required=True)

    val_b = validate_sub.add_parser("bolshoi", help="Bolshoi (Halotools) ground-truth validator")
    val_b.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    val_b.add_argument("--allow-network", action="store_true")
    val_b.add_argument("--max-n", type=int, default=5000)
    val_b.add_argument("--k", type=int, default=10)
    val_b.add_argument("--n-modes", type=int, default=10)
    val_b.add_argument("--seed", type=int, default=42)

    val_t = validate_sub.add_parser("tng", help="IllustrisTNG ground-truth validator")
    val_t.add_argument("--output-dir", type=Path, default=Path("validation_results"))
    val_t.add_argument(
        "--base-path",
        type=Path,
        default=Path("./TNG300-1/output"),
        help="Path to local TNG dataset root",
    )
    val_t.add_argument("--snapshot", type=int, default=99)
    val_t.add_argument("--min-stellar-mass", type=float, default=1e9)
    val_t.add_argument("--max-n", type=int, default=5000)
    val_t.add_argument("--k", type=int, default=10)
    val_t.add_argument("--n-modes", type=int, default=10)
    val_t.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.cmd == "operational":
        pipeline = StreamlinedFCIPipeline(k_neighbors=args.k, seed=args.seed)
        results = pipeline.run(n_galaxies=args.n_galaxies)
        fci = results["fci"]
        print(f"FCI (normalized): {fci.fci_normalized:.6f}")
        print(f"Regime: {fci.physical_regime}")
        return 0

    if args.cmd == "syk-collatz":
        r = compute_syk_collatz_fci_constant(
            N=args.N,
            kurt_collatz=args.kurt_collatz,
            kurt_syk=args.kurt_syk,
            method=args.method,
        )
        print(f"C: {r.C:.6f}")
        return 0

    if args.cmd == "meta":
        if args.meta_cmd == "audit":
            status, matched = keyword_risk_audit(args.text)
            print(status)
            if matched:
                print("matched=" + ",".join(matched))
            return 0

        if args.meta_cmd == "combine":
            s = combine_evidence_heuristic(args.evidences, prior=args.prior)
            print(f"score: {s:.6f}")
            return 0

        if args.meta_cmd == "ablation":
            import numpy as np

            x = np.asarray(args.x, dtype=np.float64)
            y = np.asarray(args.y, dtype=np.float64)
            r = ablation_sensitivity(x, y, n_trials=args.n_trials, method=args.method, seed=args.seed)
            print(json.dumps(r.__dict__, indent=2, sort_keys=True))
            return 0

    if args.cmd == "tier1":
        if args.tier1_cmd == "extract-radec":
            out = extract_radec_to_npz(
                catalog_path=args.catalog,
                out_npz=args.out_npz,
                ra_col=args.ra_col,
                dec_col=args.dec_col,
                max_rows=args.max_rows,
            )
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        if args.tier1_cmd == "add-quantum-mass":
            out = add_quantum_mass_to_catalog_npz(
                catalog_npz=args.input_npz,
                out_npz=args.out_npz,
                k=int(args.k),
                n_modes=int(args.n_modes),
                threshold=float(args.threshold),
                force=bool(args.force),
            )
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        if args.tier1_cmd == "score-single":
            report = score_single_artifact(
                prediction_card_json=args.prediction_card_json,
                artifact_json=args.artifact_json,
            )
            args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(str(args.out_json))
            return 0

        if args.tier1_cmd == "score-frontier":
            th = load_thresholds(args.prediction_card)
            summary = score_frontier_manifest(manifest_path=args.frontier_manifest, thresholds=th)
            args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(str(args.out_json))
            return 0

    if args.cmd == "tier2":
        if args.tier2_cmd == "universality-sweep":
            from .tier2.universality_sweep import run_universality_sweep

            run_dir = default_run_dir(args.output_dir)
            run_universality_sweep(
                output_dir=run_dir,
                inputs=list(args.inputs),
                threshold=float(args.threshold),
                plot=bool(args.plot),
            )
            out_path = run_dir / "tier2_path1_universality_manifest.json"
            print(str(out_path))
            return 0

        if args.tier2_cmd == "stopping-time":
            from .tier2.stopping_time_vs_mass import run_stopping_time_vs_mass

            run_dir = default_run_dir(args.output_dir)
            run_stopping_time_vs_mass(
                output_dir=run_dir,
                input_npz=args.input,
                sample_size=int(args.sample_size),
                max_steps=int(args.max_steps),
                seed=int(args.seed),
                bootstrap=bool(args.bootstrap),
                bootstrap_iters=int(args.bootstrap_iters),
            )
            out_path = run_dir / "tier2_path3_stopping_time_vs_mass_manifest.json"
            print(str(out_path))
            return 0

        if args.tier2_cmd == "fingerprint":
            from .tier2.cosmic_vs_collatz_fingerprint import run_cosmic_vs_collatz_fingerprint

            run_dir = default_run_dir(args.output_dir)
            run_cosmic_vs_collatz_fingerprint(
                output_dir=run_dir,
                collatz_npz=args.collatz,
                smacs_npz=args.smacs,
                flamingo_npz=args.flamingo,
                threshold=float(args.threshold),
                k=int(args.k),
                n_modes=int(args.n_modes),
            )
            out_path = run_dir / "tier2_path2_fingerprint_manifest.json"
            print(str(out_path))
            return 0

        if args.tier2_cmd == "collatz-summary":
            from .tier2.collatz_run_summary import run_collatz_run_summary

            run_dir = default_run_dir(args.output_dir)
            run_collatz_run_summary(
                output_dir=run_dir,
                collatz_npz=args.collatz,
                baseline_npz=args.baseline,
                threshold=float(args.threshold),
                cosmic_target=float(args.cosmic_target),
                runtime_seconds=(None if args.runtime_seconds is None else float(args.runtime_seconds)),
            )
            out_path = run_dir / "tier2_collatz_run_summary_manifest.json"
            print(str(out_path))
            return 0

        if args.tier2_cmd == "plot-16m":
            from .tier2.plot_16m_discovery import run_plot_16m_discovery

            run_dir = default_run_dir(args.output_dir)
            run_plot_16m_discovery(
                output_dir=run_dir,
                collatz_npz=args.collatz,
                threshold=float(args.threshold),
                cosmic_target=float(args.cosmic_target),
            )
            out_path = run_dir / "tier2_plot_16m_discovery_manifest.json"
            print(str(out_path))
            return 0

    if args.cmd == "gate":
        issues: list[str] = []
        if args.frontier_manifest.exists():
            issues.extend(check_frontier_manifest(args.frontier_manifest))
            print("frontier: OK" if not any(s.startswith("frontier") for s in issues) else "frontier: FAIL")
        else:
            print("frontier: SKIP (manifest not found)")

        if args.universality_manifest.exists():
            issues.extend(check_universality_manifest(args.universality_manifest))
            print("universality: OK" if not any(s.startswith("universality") for s in issues) else "universality: FAIL")
        else:
            print("universality: SKIP (manifest not found)")

        if issues:
            print("\nISSUES:")
            for s in issues:
                print("- " + s)
            return 1

        print("\nGATE: PASS")
        return 0

    if args.cmd == "suite":
        if args.suite_cmd == "run":
            run_dir = default_run_dir(args.output_dir)
            manifest = run_public_suite(
                run_dir=run_dir,
                profile=args.profile,
                allow_network=bool(args.allow_network),
                frontier_clusters_json=args.frontier_clusters_json,
                tng_base_path=args.tng_base_path,
                skip_tng=bool(args.skip_tng),
                operational_n_galaxies=args.n_galaxies,
                operational_k=args.k,
                operational_seed=args.seed,
                operational_n_eigenstates=args.n_eigenstates,
                syk_N=args.syk_N,
                syk_kurt_collatz=args.kurt_collatz,
                syk_kurt_syk=args.kurt_syk,
                syk_method=args.method,
                frontier_manifest=args.frontier_manifest,
                universality_manifest=args.universality_manifest,
            )
            out_path = Path(manifest["run_dir"]) / "suite_manifest.json"
            print(str(out_path))
            return 0

        if args.suite_cmd == "universality":
            from .suites.universality import run_universality_ground_truth_suite

            run_dir = default_run_dir(args.output_dir)
            run_universality_ground_truth_suite(
                output_dir=run_dir,
                allow_network=bool(args.allow_network),
                tng_base_path=args.tng_base_path,
                k=args.k,
                n_modes=args.n_modes,
                bolshoi_max_n=args.bolshoi_max_n,
                skip_tng=bool(args.skip_tng),
            )
            out_path = run_dir / "universality_ground_truth_suite_manifest.json"
            print(str(out_path))
            return 0

        if args.suite_cmd == "frontier":
            from .suites.frontier import run_frontier_evidence_suite

            run_dir = default_run_dir(args.output_dir)
            run_frontier_evidence_suite(
                output_dir=run_dir,
                clusters_json=args.clusters_json,
                k=int(args.k),
                n_modes=int(args.n_modes),
                n_perm=int(args.n_perm),
                seed=int(args.seed),
            )
            out_path = run_dir / "frontier_evidence_suite_manifest.json"
            print(str(out_path))
            return 0

    if args.cmd == "validate":
        if args.validate_cmd == "bolshoi":
            from .validators.bolshoi import run_bolshoi_ground_truth

            run_dir = default_run_dir(args.output_dir)
            run_bolshoi_ground_truth(
                output_dir=run_dir,
                max_n=int(args.max_n),
                k=int(args.k),
                n_modes=int(args.n_modes),
                seed=int(args.seed),
                allow_network=bool(args.allow_network),
            )
            out_path = run_dir / "halotools_bolshoi_ground_truth_manifest.json"
            print(str(out_path))
            return 0

        if args.validate_cmd == "tng":
            from .validators.tng import run_tng_ground_truth

            run_dir = default_run_dir(args.output_dir)
            run_tng_ground_truth(
                base_path=args.base_path,
                output_dir=run_dir,
                snapshot=int(args.snapshot),
                min_stellar_mass=float(args.min_stellar_mass),
                max_n=int(args.max_n),
                k=int(args.k),
                n_modes=int(args.n_modes),
                seed=int(args.seed),
            )
            out_path = run_dir / "tng_validation_manifest.json"
            print(str(out_path))
            return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
