"""
rvc_extract_features.py
Replacement for RVC's extract_feature_print.py.
Uses transformers HuBERT instead of fairseq (fairseq breaks on Python 3.12).
Args: exp_dir
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import soundfile as sf
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

exp_dir = sys.argv[1]
wav_dir = os.path.join(exp_dir, '1_16k_wavs')
out_dir = os.path.join(exp_dir, '3_feature768')
os.makedirs(out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_half = torch.cuda.is_available()

print(f'Loading HuBERT (device={device}, half={is_half})...')
extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
model = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(device)
if is_half:
    model = model.half()
model.eval()
print('HuBERT loaded.')

wav_files = sorted(f for f in os.listdir(wav_dir) if f.endswith('.wav'))
print(f'Processing {len(wav_files)} files -> {out_dir}')

for i, fname in enumerate(wav_files):
    out_path = os.path.join(out_dir, fname.replace('.wav', '.npy'))
    if os.path.exists(out_path):
        continue

    audio, sr = sf.read(os.path.join(wav_dir, fname))
    audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    inputs = extractor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
    with torch.no_grad():
        vals = inputs.input_values.to(device)
        if is_half:
            vals = vals.half()
        outputs = model(vals, output_hidden_states=True)
        feats = outputs.hidden_states[9][0].float().cpu().numpy()

    np.save(out_path, feats)
    if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
        print(f'  [{i+1}/{len(wav_files)}] done')

print(f'Feature extraction complete.')
