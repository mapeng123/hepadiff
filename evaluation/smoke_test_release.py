#!/usr/bin/env python
"""Load every released asset and run minimal end-to-end inference checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hepadiff.chemical import predict_dili, predict_spectrum
from hepadiff.zero_shot import ExternalZeroShot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    parser.add_argument('--zero-shot-assets', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--encoder-path', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    assets = Path(args.assets)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    result = {'device': device, 'checks': {}}

    core = torch.load(assets / 'hepadiff_step155000.pt', map_location='cpu', weights_only=False)
    assert int(core['step']) == 155000
    assert len(core['pert_vocab']) == 10949
    result['checks']['core'] = {'step': int(core['step']), 'n_perturbations': len(core['pert_vocab'])}
    del core

    for task in ('all', 'vmost'):
        name = f'dili_{task}_ecfp4_final.pt'
        pred = predict_dili(smiles, assets / name, device)
        assert 0 <= pred['score'] <= 1 and np.isfinite(pred['score'])
        result['checks'][name] = {'score': pred['score'], 'task': pred['task']}

    for dose in ('low', 'middle', 'high'):
        meta, frame = predict_spectrum(smiles, assets / 'tggates_spectrum_chemberta_final.pt',
                                       'tggates', dose, device, args.encoder_path)
        assert frame.shape == (16000, 2) and np.isfinite(frame.predicted_response).all()
        result['checks'][f'tggates_{dose}'] = {'shape': list(frame.shape), 'dose': meta['dose']}

    for name in ('lincs_model_final_alldata.pt', 'lincs_model_seed0.pt',
                 'lincs_model_seed1.pt', 'lincs_model_seed2.pt'):
        meta, frame = predict_spectrum(smiles, assets / name, 'lincs', device=device,
                                       encoder_path=args.encoder_path)
        assert frame.shape == (978, 2) and np.isfinite(frame.predicted_response).all()
        result['checks'][name] = {'shape': list(frame.shape), 'task': meta['task']}

    checkpoint = torch.load(assets / 'hepadiff_step155000.pt', map_location='cpu', weights_only=False)
    zero = ExternalZeroShot(args.zero_shot_assets, checkpoint, args.reference)
    info = zero.info()
    assert info['n_supported_unseen_targets'] == 7050 and info['n_unseen_panel_targets'] == 7130
    status = zero.check_gene('ABCB11')
    assert status['external_supported'] and status['route'] == 'external_zero_shot_then_core_mapping'
    metadata, frame, arrays = zero.predict('ABCB11', 'hepatocyte')
    assert frame.shape == (16000, 5) and np.isfinite(arrays['mapped_cell_state']).all()
    result['checks']['zero_shot'] = {'coverage': info, 'test_gene': status,
                                     'n_nonzero': metadata['n_nonzero_external_response']}

    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('ALL RELEASE SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
