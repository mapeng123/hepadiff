#!/usr/bin/env python
"""Train deployment refits of the DILI and TG-GATEs chemical interfaces.

These all-data models are for inference only. Manuscript performance remains
based on the independent held-out evaluation runs, not on these refits.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Spectrum(nn.Module):
    def __init__(self, din, genes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.15), nn.Linear(1024, genes))

    def forward(self, x):
        return self.net(x)


def dili_one(task, features, labels, device, out):
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).float().to(device)
    torch.manual_seed(100)
    model = Classifier(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    pos_w = torch.tensor((labels == 0).sum() / max((labels == 1).sum(), 1)).float().to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 128):
            b = perm[start:start + 128]
            loss = lossf(model(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 50 == 49:
            print(f'DILI {task} epoch {epoch + 1}/200 loss={loss.item():.5f}', flush=True)
    model.eval()
    target = 'combined_DILI_concern_vs_no_concern' if task == 'all' else 'vMost_vs_no_concern'
    checkpoint = {
        'format': 'hepadiff_interface_v1', 'task': target, 'feature': 'ECFP4',
        'input_dim': 2048, 'output_dim': 1, 'model_state': model.state_dict(),
        'training_scope': 'all eligible DILIrank v2 compounds; deployment refit, not held-out evaluation',
        'seed': 100, 'epochs': 200, 'n_compounds': int(len(labels)),
        'n_positive': int(labels.sum()), 'n_negative': int((labels == 0).sum()),
        'output': 'sigmoid score for research triage; not clinical severity or a calibrated probability',
    }
    path = out / f'dili_{task}_ecfp4_final.pt'
    torch.save(checkpoint, path)
    print('SAVED', path, path.stat().st_size, flush=True)


def tggates(data, device, out):
    logfc = data['tggates_logfc'].astype(np.float32)
    emb = data['tggates_emb'].astype(np.float32)
    doses = [str(x) for x in data['tggates_doses']]
    compounds = [str(x) for x in data['tggates_compounds']]
    genes = [str(x) for x in data['genes']]
    if emb.shape != (119, 384) or logfc.shape != (119, 3, 16000) or len(genes) != 16000:
        raise ValueError(f'Unexpected shapes: {emb.shape}, {logfc.shape}, genes={len(genes)}')
    xs, ys = [], []
    for ci in range(len(compounds)):
        for di in range(len(doses)):
            dose = np.zeros(len(doses), dtype=np.float32)
            dose[di] = 1
            xs.append(np.concatenate([emb[ci], dose]))
            ys.append(logfc[ci, di])
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys)).to(device)
    torch.manual_seed(100)
    model = Spectrum(x.shape[1], len(genes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 64):
            b = perm[start:start + 64]
            pred = model(x[b])
            mse = ((pred - y[b]) ** 2).mean()
            cos = 1 - torch.nn.functional.cosine_similarity(pred, y[b], dim=-1).mean()
            loss = mse + 0.3 * cos
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 50 == 49:
            print(f'TG-GATEs epoch {epoch + 1}/300 loss={loss.item():.5f}', flush=True)
    model.eval()
    checkpoint = {
        'format': 'hepadiff_interface_v1', 'task': 'TG-GATEs_dose_resolved_transcriptional_spectrum',
        'feature': 'ChemBERTa-2-77M-MTR-frozen-mean-pooled',
        'chemberta_model': 'DeepChem/ChemBERTa-77M-MTR',
        'chemberta_revision': '66b895cab8adebea0cb59a8effa66b2020f204ca',
        'input_dim': 387, 'embedding_dim': 384, 'output_dim': len(genes),
        'output_genes': genes, 'dose_levels': doses, 'model_state': model.state_dict(),
        'training_scope': 'all 119 TG-GATEs compounds x three doses; deployment refit, not held-out evaluation',
        'seed': 100, 'epochs': 300, 'n_compounds': len(compounds),
        'output': '16000-gene log-fold-change-like response relative to matched controls',
    }
    path = out / 'tggates_spectrum_chemberta_final.pt'
    torch.save(checkpoint, path)
    print('SAVED', path, path.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, flush=True)
    data = np.load(args.data, allow_pickle=False)
    start = time.time()
    for task in ('all', 'vmost'):
        dili_one(task, data[f'dili_{task}_x'], data[f'dili_{task}_y'], device, out)
    tggates(data, device, out)
    provenance = {
        'purpose': 'deployment-only all-data refits', 'evaluation_claims_changed': False,
        'source_data': ['DILIrank v2', 'TG-GATEs'], 'seed': 100,
        'device': device, 'elapsed_seconds': time.time() - start,
    }
    (out / 'training_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'COMPLETE in {time.time() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
