"""PPI/GenePT external zero-shot KO responses mapped to liver reference states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class ExternalZeroShot:
    def __init__(self, assets: str | Path, checkpoint: dict, reference: str | Path):
        self.assets = Path(assets)
        self.manifest = json.loads((self.assets / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("method") != "joint_raw_pool":
            raise ValueError("Unsupported external zero-shot asset manifest")
        parameters = (self.manifest.get("genept_neighbors"), self.manifest.get("beta_genept"),
                      self.manifest.get("alpha_shrinkage"))
        if parameters != (100, 0.5, 0.45):
            raise ValueError(f"Unexpected external zero-shot parameters: {parameters}")
        self.reference = np.load(reference, allow_pickle=True)
        self.genes = [str(x) for x in self.reference["genes"]]
        self.cell_types = [str(x) for x in self.reference["celltypes"]]
        if self.manifest.get("response_genes") != self.genes:
            raise ValueError("External response genes do not match the release liver reference")
        self.query_index = {g: i for i, g in enumerate(self.manifest["query_targets"])}
        self.profiles = np.load(self.assets / "profiles.npy", mmap_mode="r")
        self.support = np.load(self.assets / "support.npy")
        if self.profiles.shape != (len(self.query_index), len(self.genes)):
            raise ValueError(f"Unexpected external profile shape: {self.profiles.shape}")
        perturbations = set(checkpoint.get("pert_vocab", {}))
        self.trained = perturbations.intersection(self.genes)
        self.step = int(checkpoint.get("step", 155000))
        if self.step != 155000:
            raise ValueError("External zero-shot assets are calibrated for the step-155000 reference")
        self.method_id = hashlib.sha256((self.assets / "manifest.json").read_bytes()).hexdigest()[:16]

    def check_gene(self, gene: str) -> dict:
        gene = gene.strip().upper()
        index = self.query_index.get(gene)
        supported = bool(gene in self.genes and index is not None and
                         np.isfinite(self.support[index]) and self.support[index] > 0)
        native = gene in self.trained
        route = "native_core" if native else ("external_zero_shot_then_core_mapping" if supported else "ABSTAIN")
        return {"gene": gene, "in_panel": gene in self.genes, "native_trained": native,
                "external_supported": supported, "knockout_supported": bool(native or supported),
                "route": route, "checkpoint_step": self.step, "external_method": "joint_raw_pool"}

    def predict(self, gene: str, cell_type: str = "hepatocyte") -> tuple[dict, pd.DataFrame, dict]:
        status = self.check_gene(gene)
        if not status["in_panel"] or not status["external_supported"]:
            reason = "outside the 16000-gene response panel" if not status["in_panel"] else "no PPI/GenePT neighbor support"
            raise ValueError(f"ABSTAIN: {status['gene']}: {reason}")
        if status["native_trained"]:
            raise ValueError(f"{status['gene']} is in the native trained vocabulary; use 'hepadiff infer --ko {status['gene']}'")
        aliases = {"hsc": "hepatic stellate cell", "kupffer": "Kupffer cell",
                   "lsec": "endothelial cell of hepatic sinusoid"}
        cell_type = aliases.get(cell_type.lower(), cell_type)
        if cell_type not in self.cell_types:
            raise ValueError(f"Unsupported liver reference cell type: {cell_type}")
        raw = np.asarray(self.profiles[self.query_index[status["gene"]]], dtype=np.float32).copy()
        base = np.asarray(self.reference[f"Ec__{cell_type}"], dtype=np.float32)
        state = np.maximum(base + raw, 0.0)
        mapped = state - base
        if not np.isfinite(raw).all() or not np.isfinite(state).all() or state.shape != (16000,):
            raise RuntimeError("External zero-shot mapping produced invalid values")
        frame = pd.DataFrame({"gene": self.genes, "external_delta": raw,
                              "reference": base, "mapped_cell_state": state, "mapped_delta": mapped})
        metadata = {
            "status": "predicted", "gene": status["gene"], "cell_type": cell_type,
            "route": "external_zero_shot_then_core_mapping", "checkpoint_step": self.step,
            "external_method": "joint_raw_pool", "model_manifest_id": self.method_id,
            "sources": self.manifest.get("external_source", []),
            "parameters": {"genept_k": 100, "beta": 0.5, "alpha": 0.45},
            "response_genes_aligned": 16000, "neighbor_support": float(self.support[self.query_index[status["gene"]]]),
            "n_nonzero_external_response": int(np.count_nonzero(raw)),
            "evidence_scope": "External pooled response mapped into a liver reference state; cell type supplies baseline context, not a separately learned cell-type-specific KO interaction.",
            "validation_scope": "Target-held-out experimental contexts; not a measured primary-liver KO validation for every query.",
        }
        arrays = {"genes": np.array(self.genes), "reference": base, "external_delta": raw,
                  "mapped_cell_state": state, "mapped_delta": mapped}
        return metadata, frame, arrays

    def info(self) -> dict:
        unseen = set(self.genes) - self.trained
        supported = [g for g in unseen if g in self.query_index and self.support[self.query_index[g]] > 0]
        return {"checkpoint_step": self.step, "external_method": "joint_raw_pool",
                "n_panel_genes": len(self.genes), "n_native_perturbation_targets": len(self.trained),
                "n_unseen_panel_targets": len(unseen), "n_supported_unseen_targets": len(supported),
                "n_unsupported_unseen_targets": len(unseen) - len(supported),
                "model_manifest_id": self.method_id, "cell_type_specific_perturbation_validation": False}
