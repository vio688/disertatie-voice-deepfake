# Neural Voice Identity Control & Deepfake Audio Analysis

> **Proiect academic — disertație 2026.**
> Utilizat exclusiv pentru demonstrație academică. Nu pentru uz public sau comercial.

## Ce face sistemul

| Funcționalitate | Descriere |
|---|---|
| **Voice Conversion** | Transformă timbrul vocal al unui fișier audio în vocea uneia din 3 persoane publice românești, păstrând conținutul lingvistic |
| **Deepfake Detection** | Analizează un fișier audio și produce: scor global de probabilitate AI-generated + timeline temporal + spectrogramă cu overlay |

## Cum rulezi demo-ul (Google Colab)

1. Deschide `notebooks/03_main_app.ipynb` în Google Colab
2. Conectează Google Drive (celula de mount)
3. Rulează toate celulele (`Runtime → Run all`)
4. Accesează link-ul Gradio generat (format: `https://xxx.gradio.live`)

> **Cerință:** Modelele fine-tuned (`voice_1.pth`, `voice_2.pth`, `voice_3.pth`) trebuie să fie deja pe Google Drive la calea configurată. Rulează `02_rvc_finetune.ipynb` prima dată.

## Structura repo

```
├── notebooks/
│   ├── 01_data_collection.ipynb   # Descărcare audio + preprocesare (yt-dlp + demucs)
│   ├── 02_rvc_finetune.ipynb      # Fine-tuning RVC v2 per voce
│   └── 03_main_app.ipynb          # Aplicația Gradio principală (demo)
├── scripts/
│   ├── data_collection.py         # Script standalone colectare date
│   ├── preprocess_audio.py        # Trim silence, normalize, split clipuri
│   ├── voice_convert.py           # Funcție inferență VC (wrapper RVC)
│   └── deepfake_detect.py         # Funcție inferență deepfake (HuggingFace)
├── data/                          # gitignored — audio raw și procesat
├── models/checkpoints/            # gitignored — checkpoint-uri .pth
├── docs/                          # Figuri, diagrame, material susținere
├── Memory/                        # Vault Obsidian — memoria persistentă proiect
└── requirements.txt
```

## Tehnologii

- **Voice Conversion:** RVC v2 (fine-tune per voce, ~1-2h pe Colab T4)
- **Deepfake Detection:** `MelodyMachine/Deepfake-audio-detection-V2` (HuggingFace, fără fine-tune)
- **UI:** Gradio 4.x (3 tab-uri: VC, Deepfake, About)
- **Mediu:** Google Colab free tier (T4 GPU) + Google Drive pentru persistență

## Disclaimer

Vocile folosite sunt persoane publice ale căror înregistrări sunt disponibile public online (interviuri, podcast-uri, discursuri). Sistemul este construit exclusiv în scop academic, pentru demonstrarea capacităților și limitelor tehnologiilor de sinteză vocală și detecție deepfake. Nu este destinat înșelăciunii, uzului comercial sau distribuției publice.
