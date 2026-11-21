"""Inference for HepaDiff chemical-structure interfaces."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"


class DILIClassifier(nn.Module):
    def __init__(self, input_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SpectrumMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.15), nn.Linear(1024, output_dim))

    def forward(self, x):
        return self.net(x)


def ecfp4(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("DILI inference requires RDKit: python -m pip install rdkit") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fingerprint, dtype=np.float32)


def chemberta_embedding(smiles: str, device: str, encoder_path: str | None = None) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Spectrum inference requires transformers: python -m pip install transformers") from exc
    source = encoder_path or CHEMBERTA_MODEL
    kwargs = {} if encoder_path else {"revision": CHEMBERTA_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    encoder = AutoModel.from_pretrained(source, **kwargs).to(device).eval()
    with torch.no_grad():
        tokens = tokenizer([smiles], return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        embedding = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()[0]
    return embedding.astype(np.float32)


def load_checkpoint(path: str | Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def predict_dili(smiles: str, weights: str | Path, device: str = "cpu") -> dict:
    checkpoint = load_checkpoint(weights)
    if checkpoint.get("feature") != "ECFP4" or checkpoint.get("output_dim") != 1:
        raise ValueError("This is not a released ECFP4 DILI checkpoint")
    model = DILIClassifier(checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature = torch.from_numpy(ecfp4(smiles)[None]).to(device)
    with torch.no_grad():
        score = float(torch.sigmoid(model(feature))[0].cpu())
    return {
        "interface": "DILI", "task": checkpoint["task"], "smiles": smiles,
        "score": score, "weights": str(weights),
        "training_scope": checkpoint["training_scope"],
        "interpretation": "Research-use DILI-liability score; not clinical severity or a calibrated probability.",
    }


def lincs_genes() -> list[str]:
    path = files("hepadiff").joinpath("data", "lincs978_genes.json")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_spectrum(smiles: str, weights: str | Path, interface: str,
                     dose: str = "high", device: str = "cpu",
                     encoder_path: str | None = None) -> tuple[dict, pd.DataFrame]:
    raw = load_checkpoint(weights)
    is_wrapped = isinstance(raw, dict) and "model_state" in raw
    checkpoint = raw if is_wrapped else None
    state = raw["model_state"] if is_wrapped else raw
    output_dim = int(state["net.7.weight"].shape[0])
    if interface == "lincs":
        if output_dim != 978:
            raise ValueError(f"LINCS checkpoint must have 978 outputs, found {output_dim}")
        genes = lincs_genes()
        embedding = chemberta_embedding(smiles, device, encoder_path)
        feature = embedding
        task = "LINCS_L1000_landmark_transcriptional_spectrum"
        scope = "all-data deployment refit" if not is_wrapped else checkpoint.get("training_scope", "evaluation model")
    elif interface == "tggates":
        if not is_wrapped or output_dim != 16000:
            raise ValueError("TG-GATEs requires the released wrapped 16000-gene checkpoint")
        genes = [str(x) for x in checkpoint["output_genes"]]
        doses = [str(x) for x in checkpoint["dose_levels"]]
        dose_lookup = {x.lower(): i for i, x in enumerate(doses)}
        if dose.lower() not in dose_lookup:
            raise ValueError(f"Unsupported dose '{dose}'; choose from {doses}")
        dose = doses[dose_lookup[dose.lower()]]
        embedding = chemberta_embedding(smiles, device, encoder_path)
        dose_onehot = np.zeros(len(doses), dtype=np.float32)
        dose_onehot[dose_lookup[dose.lower()]] = 1
        feature = np.concatenate([embedding, dose_onehot])
        task = checkpoint["task"]
        scope = checkpoint["training_scope"]
    else:
        raise ValueError("Spectrum interface must be 'lincs' or 'tggates'")
    model = SpectrumMLP(len(feature), output_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(feature[None]).to(device))[0].cpu().numpy()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Model produced non-finite predictions")
    frame = pd.DataFrame({"gene": genes, "predicted_response": prediction})
    frame = frame.reindex(frame.predicted_response.abs().sort_values(ascending=False).index)
    metadata = {
        "interface": interface, "task": task, "smiles": smiles, "dose": dose if interface == "tggates" else None,
        "weights": str(weights), "n_genes": output_dim, "training_scope": scope,
        "chemberta_model": CHEMBERTA_MODEL, "chemberta_revision": CHEMBERTA_REVISION,
        "interpretation": "Structure-predicted transcriptional response for research triage, not proof of efficacy or mechanism.",
    }
    return metadata, frame
