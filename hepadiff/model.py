"""HepaDiff masked discrete-diffusion model and release decoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

D_MODEL = 512
N_HEAD = 8
N_LAYER = 12
CHUNK = 48
MASK_TOKEN = 51
GID_DIM = 256


class PMDDM(nn.Module):
    """Perturbation-conditioned masked discrete-diffusion transformer."""

    def __init__(self, n_genes: int, n_cell_types: int, n_conditions: int,
                 n_perturbations: int, n_species: int, n_perturbation_types: int = 4):
        super().__init__()
        self.g = n_genes
        self.n_chunk = (n_genes + CHUNK - 1) // CHUNK
        self.g_pad = self.n_chunk * CHUNK
        self.null_pert = n_perturbations - 1

        self.gene_emb = nn.Embedding(52, D_MODEL)
        self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_chunk + 5, D_MODEL) * 0.02)
        self.ct_emb = nn.Embedding(n_cell_types, D_MODEL)
        self.cond_emb = nn.Embedding(n_conditions, D_MODEL)
        self.sp_emb = nn.Embedding(n_species, D_MODEL)
        self.ptype_emb = nn.Embedding(n_perturbation_types, GID_DIM)
        self.gene_id_emb = nn.Embedding(n_genes, GID_DIM)
        self.pert_gene_mlp = nn.Sequential(
            nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.pert_free_emb = nn.Embedding(n_perturbations, GID_DIM)
        self.pert_free_mlp = nn.Sequential(
            nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.delta_head = nn.Sequential(
            nn.Linear(D_MODEL * 3, 512), nn.GELU(), nn.Linear(512, 512),
            nn.GELU(), nn.Linear(512, n_genes))
        self.z0_proj = nn.Linear(D_MODEL, D_MODEL)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                D_MODEL, N_HEAD, D_MODEL * 4, batch_first=True,
                norm_first=True, activation="gelu"), N_LAYER)
        self.decompress = nn.Linear(D_MODEL, CHUNK * 52)

    def z0_token(self, x0):
        e = self.gene_emb(x0.long()).view(x0.shape[0], self.n_chunk, CHUNK * D_MODEL)
        return self.z0_proj(self.compress(e).mean(1))

    def pert_vector(self, pert, pgid, ptype):
        v = torch.zeros(pert.shape[0], D_MODEL, device=pert.device)
        is_null = pert.eq(self.null_pert)
        pt = self.ptype_emb(ptype.clamp(0, 3))
        gene_mask = (pgid >= 0) & ~is_null
        if gene_mask.any():
            v[gene_mask] = self.pert_gene_mlp(
                self.gene_id_emb(pgid[gene_mask]) + pt[gene_mask])
        free_mask = (pgid < 0) & ~is_null
        if free_mask.any():
            v[free_mask] = self.pert_free_mlp(
                self.pert_free_emb(pert[free_mask]) + pt[free_mask])
        return v

    def forward(self, x, ct, cond, pert, z0, pgid, ptype, sp):
        batch = x.shape[0]
        e = self.gene_emb(x.long()).view(batch, self.n_chunk, CHUNK * D_MODEL)
        z = self.compress(e)
        pv = self.pert_vector(pert, pgid, ptype)
        head = torch.stack(
            [self.ct_emb(ct), self.cond_emb(cond), pv, z0, self.sp_emb(sp)], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 5:]
        return self.decompress(z).view(batch, self.g_pad, 52)


def load_model(checkpoint_path: str, n_genes: int, device: str | torch.device):
    """Load a HepaDiff checkpoint with strict state-dict validation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ct_vocab = checkpoint["ct_vocab"]
    cond_vocab = checkpoint["cond_vocab"]
    pert_vocab = checkpoint["pert_vocab"]
    species_vocab = checkpoint.get(
        "species_vocab", {"human": 0, "mouse": 1, "rat": 2, "cellline": 3})
    model = PMDDM(
        n_genes=n_genes,
        n_cell_types=len(ct_vocab) + 1,
        n_conditions=len(cond_vocab) + 1,
        n_perturbations=len(pert_vocab) + 1,
        n_species=len(species_vocab) + 1,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def decode(model: PMDDM, *, cell_type_id: int, condition_id: int,
           perturbation_id: int, gene_id: int, perturbation_type_id: int,
           species_id: int, n_cells: int, seed: int = 0, batch_size: int = 250,
           steps: int = 4, device: str | torch.device | None = None) -> np.ndarray:
    """Generate cells using the release four-step, unbiased-logit decoder."""
    if steps < 1:
        raise ValueError("steps must be positive")
    device = device or next(model.parameters()).device
    outputs = []
    for start in range(0, n_cells, batch_size):
        torch.manual_seed(seed + start)
        batch = min(batch_size, n_cells - start)
        ct = torch.full((batch,), cell_type_id, dtype=torch.long, device=device)
        cond = torch.full((batch,), condition_id, dtype=torch.long, device=device)
        pert = torch.full((batch,), perturbation_id, dtype=torch.long, device=device)
        pgid = torch.full((batch,), gene_id, dtype=torch.long, device=device)
        ptype = torch.full((batch,), perturbation_type_id, dtype=torch.long, device=device)
        sp = torch.full((batch,), species_id, dtype=torch.long, device=device)
        z0 = torch.zeros(batch, D_MODEL, device=device)
        x = torch.full((batch, model.g_pad), MASK_TOKEN, dtype=torch.long, device=device)
        order = torch.argsort(torch.rand(batch, model.g_pad, device=device), dim=1)
        for step in range(steps):
            logits = model(x, ct, cond, pert, z0, pgid, ptype, sp)
            logits[..., MASK_TOKEN] = -1e9
            sampled = torch.distributions.Categorical(logits=logits).sample()
            left = int(step * model.g_pad / steps)
            right = int((step + 1) * model.g_pad / steps)
            reveal = order[:, left:right]
            x.scatter_(1, reveal, sampled.gather(1, reveal))
        x[x == MASK_TOKEN] = 0
        outputs.append(x[:, :model.g].cpu().numpy().astype(np.int8))
    return np.concatenate(outputs, axis=0)
