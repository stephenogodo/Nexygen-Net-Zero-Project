"""
Loads the models shipped by pipeline/train_pipeline.py (one winner per
scope, per models/model_manifest.json) and exposes a single, uniform
.forecast(steps) interface regardless of whether the winner is a SARIMA
(statsmodels) or Prophet model -- the two have completely different native
APIs (.forecast(steps) vs. make_future_dataframe()+predict()).
"""
import json
from pathlib import Path

from statsmodels.iolib.smpickle import load_pickle


class ModelEntry:
    def __init__(self, model_type: str, model, params: dict, metrics: dict):
        self.model_type = model_type
        self.model = model
        self.params = params
        self.metrics = metrics

    def forecast(self, steps: int) -> list[float]:
        if self.model_type == "sarima":
            pred = self.model.forecast(steps=steps)
            return [float(v) for v in pred.tolist()]
        if self.model_type == "prophet":
            future = self.model.make_future_dataframe(periods=steps, freq="MS")
            fc = self.model.predict(future)
            return [float(v) for v in fc["yhat"].tail(steps).tolist()]
        raise ValueError(f"Unknown model_type: {self.model_type!r}")


def load_models(model_dir: Path) -> tuple[dict[str, ModelEntry], dict]:
    manifest_path = model_dir / "model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    models: dict[str, ModelEntry] = {}
    for scope_key, info in manifest["scopes"].items():
        model_type = info["model_type"]
        model_path = model_dir / info["model_file"]

        if model_type == "sarima":
            model = load_pickle(str(model_path))
        elif model_type == "prophet":
            from prophet.serialize import model_from_json

            with open(model_path) as f:
                model = model_from_json(f.read())
        else:
            raise ValueError(f"Unknown model_type in manifest for {scope_key}: {model_type!r}")

        models[scope_key] = ModelEntry(model_type, model, info["params"], info["metrics"])

    return models, manifest
