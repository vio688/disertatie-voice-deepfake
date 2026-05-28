import torch, shutil
from pathlib import Path

def export_model(vid, cfg_obj, sr=40000):
    src = str(MODELS_DIR / cfg_obj['model_file'])
    tmp = f'/content/exp_{vid}.pth'
    ckpt = torch.load(src, map_location='cpu')
    keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    if 'weight' in keys and 'config' in keys:
        print(vid + ': deja OK')
        return
    if 'model' in keys:
        sd = ckpt['model']
    else:
        sd = ckpt
    weight = {}
    for k, v in sd.items():
        if 'enc_q' not in k:
            weight[k] = v.half()
    n_spk = weight['emb_g.weight'].shape[0]
    cfg_arr = [
        1025, 32,
        192, 192, 768, 2, 6, 3, 0,
        '1',
        [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [10, 10, 2, 2],
        512,
        [16, 16, 4, 4],
        n_spk,
        256,
        sr,
    ]
    opt = {
        'weight': weight,
        'config': cfg_arr,
        'info': 'exported',
        'sr': str(sr),
        'f0': 1,
        'version': 'v2',
    }
    torch.save(opt, tmp)
    shutil.copy2(tmp, src)
    check = torch.load(src, map_location='cpu')
    print(vid + ': ' + str(list(check.keys())))

for vid, cfg in VOICE_CONFIG.items():
    export_model(vid, cfg)
