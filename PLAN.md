# Plan — Neural Voice Identity Control and Deepfake Audio Analysis System

> Plan aprobat 2026-05-18. Detaliile sunt sincronizate cu vault-ul Obsidian din `Memory/`. Acest fișier este snapshot-ul final al planului inițial.

## Context

Disertație 2 săptămâni, sistem care combină:

1. **Voice Conversion (VC)** — input audio (microfon-recording sau file upload) → output audio cu același conținut, dar în vocea uneia dintre 3 persoane publice românești pre-fine-tuned.
2. **Deepfake Audio Detection** — input audio → scor global „probabilitate AI-generated" + timeline temporal cu fluctuațiile probabilității, suprapuse pe spectrogramă.

Constrângeri ferme:
- **Limba țintă:** română (în principal).
- **Hardware:** doar CPU local + Google Colab (free tier).
- **Timp:** 14 zile.
- **Voci:** 3 voci publice RO (utilizatoarea le specifică la M0), extensibil dacă rămâne timp.
- **UI:** Gradio (Python-only, lansat din Colab).
- **Workflow:** repo GitHub, notebook-uri `.ipynb` deschise în Colab direct din GitHub.
- **Document scris al disertației:** **NU** se face acum; doar implementare + explicații tehnice.

## Decizii finale (blocate)

| # | Decizie | Notă |
|---|---|---|
| D1 | **Voice Conversion** (nu TTS) | Audio → audio; păstrăm conținutul, schimbăm timbrul |
| D2 | **3 voci publice RO** în prima fază, extensibil la 5 | Utilizatoarea le specifică la kickoff |
| D3 | **RVC v2** ca model VC principal, fine-tune per voce | Standard, multe Colab-uri existente, ~1-2h/voce pe T4 |
| D4 | **Modele pre-antrenate HuggingFace** pentru deepfake | Fără fine-tune (timp insuficient) |
| D5 | **Gradio 4.x** ca frontend | UI cu 3 tab-uri: VC, Deepfake, About |
| D6 | **Google Colab + Gradio share link** ca deployment | `app.launch(share=True)` → URL `.gradio.live` public 72h |
| D7 | **GitHub repo** pentru cod, Colab încarcă direct din GitHub | Workflow: edit local → commit → Colab open from GitHub |
| D8 | **Doar file-upload în prima fază** | Microfon record dacă rămâne timp; fără streaming |
| D9 | **Deepfake: scor global + timeline + spectrogramă overlay** | Vizualizări matplotlib/plotly în Gradio |
| D10 | **Disclaimer academic ferm** pe UI | „Proiect academic, nu pentru uz public" — implicații legale voci publice |

## Arhitectură finală

```
┌─────────────────────── Google Colab Notebook ───────────────────────┐
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Gradio App (3 tab-uri)                                     │   │
│  │  ┌─────────────────┬─────────────────┬─────────────────┐    │   │
│  │  │  Tab Voice      │  Tab Deepfake   │  Tab About      │    │   │
│  │  │  Conversion     │  Detection      │  / Disclaimer   │    │   │
│  │  └────────┬────────┴────────┬────────┴─────────────────┘    │   │
│  └───────────┼─────────────────┼─────────────────────────────┘   │
│              │                 │                                  │
│  ┌───────────▼───────┐  ┌──────▼────────────────┐                │
│  │  voice_convert()  │  │  detect_deepfake()    │                │
│  │  ─────────────    │  │  ──────────────       │                │
│  │  Input: audio +   │  │  Input: audio         │                │
│  │  voice_id (3 opt) │  │  Output:              │                │
│  │  Output: audio    │  │   - global score      │                │
│  │  cu vocea aleasă  │  │   - timeline probs    │                │
│  │                   │  │   - spectrogramă      │                │
│  │  → RVC v2         │  │  → wav2vec2-AASIST    │                │
│  │    (3 checkpoints │  │    (HuggingFace       │                │
│  │     fine-tuned)   │  │     pre-trained)      │                │
│  └───────────────────┘  └───────────────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        Gradio share URL: https://xxx.gradio.live (public 72h)
```

## Stack tehnic (concret)

| Layer | Tehnologie | Versiune | Note |
|---|---|---|---|
| Limbaj | Python | 3.10+ | Standard Colab |
| VC model | RVC v2 | `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` | Repo GitHub oficial |
| Deepfake | HuggingFace pre-trained | `MelodyMachine/Deepfake-audio-detection-V2` (sau echivalent) | Fără fine-tune |
| UI | Gradio | 4.x | `gr.Audio()`, `gr.Plot()`, `gr.Tabs()` |
| Audio I/O | librosa, soundfile, ffmpeg | latest | Resampling, manipulare |
| YT scraping | yt-dlp | latest | Pentru colectare voci |
| Voice sep | Demucs | v4 | Separare voce de muzică/zgomot |
| Plotting | matplotlib + plotly | latest | Timeline + spectrogramă |
| Notebook | Jupyter `.ipynb` | — | Tip de fișier pentru Colab |
| Versionare | Git + GitHub | — | Repo public sau privat |

## Date — strategie pentru fiecare voce

Pentru fiecare din cele 3 voci publice RO:

1. **Sursă:** YouTube — interviuri, podcast-uri, discursuri (cele mai curate sunt podcast-urile audio).
2. **Cantitate țintă:** 10-30 min audio curat per voce (vorbire continuă, fără muzică, fără overlap).
3. **Pipeline preprocessing:**
   - `yt-dlp` → download audio (mp3/m4a)
   - `demucs --two-stems=vocals` → izolare voce
   - `ffmpeg` → conversie 16kHz mono WAV
   - Trim manual segmente cu zgomot / muzică reziduală / persoane multiple
   - Split în clipuri 5-10s (cerință RVC)

## Roadmap pe 14 zile (estimare realistă)

| Zi | Task | Livrabil |
|---|---|---|
| **1** | Init repo GitHub, structură foldere, primul commit cu README + notebook gol | Repo public/privat live |
| **1-2** | Utilizatoarea alege cele 3 voci; eu fac scripturi de data collection (yt-dlp + demucs) | Audio brut colectat |
| **3** | Preprocesare audio (trim, normalize, split clipuri 5-10s) | ~15-30 min curat/voce |
| **4-5** | Setup RVC pe Colab + fine-tune secvențial cele 3 voci | 3 checkpoint-uri `.pth` salvate pe Google Drive |
| **6** | Funcție `voice_convert()` + Gradio MVP cu 1 tab funcțional | Demo VC end-to-end |
| **7** | Integrare model HuggingFace deepfake + funcție `detect_deepfake()` | Scor + timeline + spectrogramă |
| **8** | Vizualizări deepfake (matplotlib/plotly), polish output grafic | Tab Deepfake funcțional |
| **9** | Integrare ambele tab-uri în Gradio app, polish UI | App unificată |
| **10** | Testare end-to-end cu sample-uri RO; fix bug-uri | App stabilă |
| **11** | (Bonus) Adăugare microfon recording în Gradio | Tab VC accept recording |
| **12** | Generare exemple demo (5-10 input/output pairs) + screenshots | Material pentru susținere |
| **13** | Buffer pentru întârzieri | — |
| **14** | Final polish + livrare | Demo live + cod pe GitHub |

**Buffer total în plan: zi 13 + ~20% slack zilnic.** Dacă fine-tune-ul depășește, fallback la **zero-shot OpenVoice v2** (fără training).

## Modele recomandate

### Voice Conversion
- **Primary:** RVC v2 — `github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
- **Fallback zero-shot:** OpenVoice v2 — `github.com/myshell-ai/OpenVoice`
- **Alternative:** Seed-VC, so-vits-svc 4.1

### Deepfake Detection (HuggingFace, pre-trained)
- `MelodyMachine/Deepfake-audio-detection-V2`
- `mo-thecreator/Deepfake-audio-detection`
- `Heem2/audio-deepfake-detection`
- Backup: AASIST originar (`clovaai/aasist`) — necesită mai mult cod

## Surse de date

- **YouTube** (cu yt-dlp) pentru voci RO publice
- **(Pentru evaluare deepfake, opțional):** ASVspoof 2021 dataset — DOAR dacă rămâne timp

## Riscuri & mitigation

| Risc | Impact | Mitigation |
|---|---|---|
| Fine-tune RVC durează > estimat | Mare | Fallback la zero-shot OpenVoice v2; reducere la 2 voci |
| Colab session timeout în training | Mediu | Checkpoint frecvent pe Google Drive; reluare de la ultimul |
| Calitate VC slabă pe RO | Mediu | Mai mult audio curat per voce; experimentare cu pitch/index ratio RVC |
| Deepfake detector generalizează prost pe RO | Mic-mediu | Folosim model robust + disclaimer că „precision pe RO nu e validată" |
| Probleme etice/legale voci publice | Mediu | Disclaimer ferm pe UI; demo doar pentru susținere; repo privat la nevoie |
| Demo Colab cade înainte de susținere | Mediu | Backup local cu screenshots + video înregistrat |

## Workflow operațional

1. Eu editez notebook-uri `.ipynb` + scripts `.py` local pe laptop
2. Git commit + push la GitHub
3. Utilizatoarea: în Colab → `File → Open notebook → GitHub → URL` → primește versiunea curentă
4. Rulează celulele în Colab; experimentele/checkpoint-urile se salvează în Google Drive montat
5. Dacă utilizatoarea modifică ceva în Colab și vrea persistent → `File → Save a copy in GitHub`
6. Eu fac `git pull` și văd modificările

## Verificare end-to-end (criterii de succes)

Sistemul e gata când:
- [ ] Gradio app pornește în Colab (≤ 5 min de la `Run All`)
- [ ] Tab VC: upload `.wav` + select voce → output audio descărcabil
- [ ] Cele 3 voci funcționează (audibil ≠ vocea inputului)
- [ ] Tab Deepfake: upload audio → scor + grafic timeline + spectrogramă
- [ ] Aplicația rulează 30 min fără crash
- [ ] Link `.gradio.live` accesibil din alt device
- [ ] Repo GitHub conține: notebook principal, scripts data, README cu instrucțiuni Colab, disclaimer

## Detalii suplimentare în vault

Memoria persistentă din `Memory/` conține detalii suplimentare pe categorii. Vezi în special:
- `Memory/Roadmap.md` — milestones detaliate
- `Memory/Architecture.md` — arhitectura finală
- `Memory/Neural_Networks.md` — modele și fine-tuning
- `Memory/Data_Pipeline.md` — preprocesare audio
- `Memory/Decision_Log.md` — toate ADR-urile cu motivație
