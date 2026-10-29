"""Command line entry point: python -m ecg.cli <command>."""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from ecg.labels.mapping import DEFAULT_LABELS, load_label_spec


def find_latest_cache(cache_dir="data/cache"):
    p = Path(cache_dir)
    dirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "manifest.json").exists()]
    if not dirs:
        raise FileNotFoundError(f"No completed dataset cache found in {cache_dir}. Run 'build-data' first.")
    return max(dirs, key=lambda d: (d / "manifest.json").stat().st_mtime)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ecg")
    sub = parser.add_subparsers(dest="command", required=True)
    
    # audit
    audit = sub.add_parser("audit", help="validate the harmonized label mapping table")
    audit.add_argument("--labels", default=str(DEFAULT_LABELS))
    
    # Step 1: build-labels
    lab = sub.add_parser("build-labels", help="step 1: label matrix + mask from PTB-XL metadata, counts printed")
    lab.add_argument("--root", default="data/raw/files/ptb-xl/1.0.3")
    lab.add_argument("--labels", default=str(DEFAULT_LABELS))
    lab.add_argument("--out", default="data/processed/ptbxl_labels_v1.npz")
    
    # Step 2: build-data
    bd = sub.add_parser("build-data", help="step 2: preprocess signals into a cached [N,12,T] tensor dataset")
    bd.add_argument("--config", default="configs/baseline.yaml")
    bd.add_argument("--limit", type=int, default=None)
    bd.add_argument("--no-hash", action="store_true", help="skip per-file SHA256 (faster)")
    
    # Step 3: baseline
    base_cmd = sub.add_parser("baseline", help="step 3: train & evaluate LightGBM baseline on handcrafted features")
    base_cmd.add_argument("--cache", default=None, help="path to cache directory (defaults to latest)")
    base_cmd.add_argument("--seed", type=int, default=42)
    base_cmd.add_argument("--out", default="runs/baseline")
    
    # Step 4: train (ResNet1D / deep model)
    tr = sub.add_parser("train", help="step 4: train deep model with masked BCE loss & tracking")
    tr.add_argument("--config", default="configs/baseline.yaml")
    tr.add_argument("--cache", default=None, help="path to cache directory (defaults to latest)")
    tr.add_argument("--seed", type=int, default=None)
    tr.add_argument("--resume", action="store_true")
    
    # Step 5: eval
    ev = sub.add_parser("eval", help="step 5: single evaluation command printing all metrics + bootstrap CIs")
    ev.add_argument("--config", default="configs/baseline.yaml")
    ev.add_argument("--checkpoint", default=None, help="path to model weights checkpoint (.pt)")
    ev.add_argument("--cache", default=None)
    ev.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ev.add_argument("--n-boot", type=int, default=1000)

    # Step 7: export bundle
    exp = sub.add_parser("export", help="step 7: calibrate model, export to ONNX, run parity test, and write release bundle")
    exp.add_argument("--config", default="configs/baseline.yaml")
    exp.add_argument("--checkpoint", default=None, help="path to model weights checkpoint (.pt)")
    exp.add_argument("--cache", default=None)
    exp.add_argument("--out", default="bundle/v1.0.0")
    
    args = parser.parse_args(argv)
    
    if args.command == "audit":
        spec = load_label_spec(args.labels)
        print(f"Label spec {spec['version']}: {len(spec['classes'])} classes, {len(spec.get('mappings', []))} reviewed mappings")
        return

    if args.command == "build-labels":
        spec = load_label_spec(args.labels)
        from ecg.datasets.label_matrix import build_label_matrix, crosscheck_mapping
        issues = crosscheck_mapping(args.root, spec)
        print(f"Mapping vs dataset statement table: {'OK' if not issues else str(len(issues)) + ' disagreement(s)'}")
        for i in issues:
            print("  -", i)
        report = build_label_matrix(args.root, spec, args.out)
        print(f"Saved {args.out}")
        print(f"Records: {report['n_records']}  labeled: {report['labeled_records']}  "
              f"unlabeled(masked): {report['unlabeled_records']}  multi-label: {report['multi_label_records']}")
        print("Positives per class:")
        for c, n in report["positives"].items():
            print(f"  {c:5s} {n:6d}  " + "  ".join(f"{s}={report['per_split'][s][c]}" for s in report["per_split"]))
        print("Split sizes:", report["split_sizes"], " patient overlap:", report["patient_overlap"])
        print("Top unmapped codes:", report["unmapped_codes"])
        Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    if args.command == "build-data":
        from ecg.config import ExperimentConfig
        from ecg.datasets.build import build_dataset, preprocess_config_from
        cfg = ExperimentConfig.from_yaml(args.config)
        spec = load_label_spec(cfg.data.labels or None)
        out = build_dataset(cfg.data.source, cfg.data.root, spec, preprocess_config_from(cfg.data), cfg.data.cache_dir,
                            cfg.data.split_seed, args.limit, hash_files=not args.no_hash)
        print(f"Cache: {out}")
        import numpy as np
        X = np.load(out / "signals.npy", mmap_mode="r")
        print(f"signals {X.shape} {X.dtype}")
        audit = json.loads((out / "audit.json").read_text())
        print("records per split:", {k: v["records"] for k, v in audit["per_split"].items()},
              " patient overlap:", audit["patient_overlap"])
        return

    if args.command == "baseline":
        from ecg.models.lgbm_baseline import train_eval_baseline
        cache = Path(args.cache) if args.cache else find_latest_cache()
        print(f"Using cache: {cache}")
        res = train_eval_baseline(cache, out_dir=args.out, seed=args.seed)
        return

    if args.command == "train":
        from ecg.config import ExperimentConfig
        from ecg.train.trainer import fit
        cfg = ExperimentConfig.from_yaml(args.config)
        spec = load_label_spec(cfg.data.labels or None)
        cache = Path(args.cache) if args.cache else find_latest_cache()
        print(f"Using cache: {cache}")
        res = fit(cfg, cache, num_classes=len(spec["classes"]), seed=args.seed, resume=args.resume)
        print(f"Training complete. Best checkpoint: {res['best_checkpoint']} (Val Macro AUROC: {res['best_val_macro_auroc']:.4f})")
        return

    if args.command == "eval":
        import torch
        from ecg.config import ExperimentConfig
        from ecg.datasets.build import ECGArrayDataset
        from ecg.models.factory import build_model
        from ecg.eval.harness import evaluate
        cfg = ExperimentConfig.from_yaml(args.config)
        spec = load_label_spec(cfg.data.labels or None)
        num_classes = len(spec["classes"])
        cache = Path(args.cache) if args.cache else find_latest_cache()
        dataset = ECGArrayDataset(cache, split=args.split)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg.model, num_classes).to(device)
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Look for best.pt in runs/
            candidates = list(Path("runs").glob("**/best.pt"))
            if not candidates:
                raise FileNotFoundError("No checkpoint specified and no runs/**/best.pt found.")
            ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        
        metrics = evaluate(model, dataset, cfg, device=device, n_boot=args.n_boot)
        print(f"\n--- Evaluation Results on '{args.split}' split ({len(dataset)} records) ---")
        print(f"Macro AUROC: {metrics['macro_auroc']:.4f}  (95% CI: {metrics['macro_auroc_ci95'][0]:.4f} - {metrics['macro_auroc_ci95'][1]:.4f})")
        print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
        if metrics.get("macro_ece") is not None:
            print(f"Macro ECE:   {metrics['macro_ece']:.4f}")
        print("\nPer-class breakdown:")
        print(f"{'Class':<8} {'AUROC':<10} {'AUPRC':<10} {'F1':<10} {'Sens':<10} {'Spec':<10} {'Prev':<10}")
        classes = [c["id"] for c in spec["classes"]]
        for k, c in enumerate(classes):
            cl = metrics["classes"][k]
            auc_s = f"{cl['auroc']:.4f}" if cl["auroc"] is not None else "N/A"
            pr_s = f"{cl['auprc']:.4f}" if cl["auprc"] is not None else "N/A"
            f1_s = f"{cl['f1']:.4f}" if cl["f1"] is not None else "N/A"
            se_s = f"{cl['sensitivity']:.4f}" if cl["sensitivity"] is not None else "N/A"
            sp_s = f"{cl['specificity']:.4f}" if cl["specificity"] is not None else "N/A"
            pv_s = f"{cl['prevalence']:.4f}" if cl["prevalence"] is not None else "N/A"
            print(f"{c:<8} {auc_s:<10} {pr_s:<10} {f1_s:<10} {se_s:<10} {sp_s:<10} {pv_s:<10}")
        return

    if args.command == "export":
        import torch
        from ecg.config import ExperimentConfig
        from ecg.models.factory import build_model
        from ecg.export.bundle import create_release_bundle
        cfg = ExperimentConfig.from_yaml(args.config)
        spec = load_label_spec(cfg.data.labels or None)
        num_classes = len(spec["classes"])
        cache = Path(args.cache) if args.cache else find_latest_cache()
        
        device = torch.device("cpu")
        model = build_model(cfg.model, num_classes).to(device)
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            candidates = list(Path("runs").glob("**/best.pt"))
            if not candidates:
                raise FileNotFoundError("No checkpoint specified and no runs/**/best.pt found.")
            ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint for export: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        
        bundle_path = create_release_bundle(model, cache, bundle_dir=args.out)
        print(f"\nRelease bundle export complete: {bundle_path}")
        return



if __name__ == "__main__":
    main()
