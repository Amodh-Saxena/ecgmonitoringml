"""Experiment tracking: local JSON always, MLflow when installed. Records provenance for every run."""
import json
import subprocess
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class RunTracker:
    def __init__(self, cfg, manifest_hash: str, root="runs", seed: int | None = None, use_mlflow: bool = True):
        self.name = cfg.run_name(seed)
        self.dir = Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info = {"run": self.name, "config_hash": cfg.config_hash(), "git_commit": git_commit(),
                     "manifest_hash": manifest_hash, "seed": cfg.train.seed if seed is None else seed,
                     "config": cfg.to_dict(), "valid": True, "invalid_reason": None}
        self.history: list[dict] = []
        self.mlflow = None
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(cfg.study)
                mlflow.start_run(run_name=self.name)
                mlflow.log_params({"config_hash": self.info["config_hash"], "git_commit": self.info["git_commit"],
                                   "manifest_hash": manifest_hash, "seed": self.info["seed"]})
                self.mlflow = mlflow
            except Exception:
                self.mlflow = None
        self._flush()

    def log_metrics(self, metrics: dict, step: int | None = None):
        self.history.append({"step": step, **metrics})
        if self.mlflow:
            self.mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
        self._flush()

    def log_artifact(self, path):
        if self.mlflow:
            self.mlflow.log_artifact(str(path))

    def mark_invalid(self, reason: str):
        self.info.update(valid=False, invalid_reason=reason)
        (self.dir / "INVALID").write_text(reason, encoding="utf-8")
        if self.mlflow:
            self.mlflow.set_tag("invalid", reason)
        self._flush()

    def _flush(self):
        (self.dir / "run.json").write_text(json.dumps({**self.info, "history": self.history}, indent=2, default=str),
                                           encoding="utf-8")

    def close(self):
        if self.mlflow:
            self.mlflow.end_run()
