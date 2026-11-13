# Abstracte plaatjes — image-node (test-app)

Eén tegel/node voor je node-based app: genereert een **abstract plaatje** met een **klein lokaal AI-model** (diffusion). Werkt op **macOS** (Apple Silicon, MPS) en **Linux** (CUDA of CPU). Geen cloud, geen API-sleutel.

## Snelstart

```bash
cd abstract-image-node
python3 -m venv .venv && source .venv/bin/activate   # op Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 app.py        # test-app met interface (tandwiel ⚙ = instellingen)
```

Alleen CLI testen (handig op een headless Linux-machine):

```bash
python3 cli.py "abstracte paarse inktvlekken" --width 768 --height 512 --seed 42
```

### macOS

- `pip install -r requirements.txt` installeert torch met ingebouwde **MPS**-ondersteuning (Apple Silicon). Device staat op `auto` → pakt automatisch `mps`.
- De GUI gebruikt `tkinter`. De systeem-`python3` van macOS heeft geen tkinter; gebruik de python van python.org of Homebrew:
  ```bash
  brew install python-tk
  /opt/homebrew/bin/python3 app.py     # Apple Silicon
  ```

### Linux

- Met NVIDIA-GPU: `pip install -r requirements.txt` pakt de CUDA-build van torch (device `auto` → `cuda`).
- Zonder GPU (CPU-only), veiliger en kleiner:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt --no-deps   # of installeer de rest los
  pip install diffusers transformers accelerate safetensors pillow
  ```
  Zet in het tandwiel-dialog `device` op `cpu`, of in `config.json`.

## Welk model?

Standaard staat in `config.json`: `segmind/tiny-sd` — een gedistilleerde Stable Diffusion (~1–2 GB download, 55% minder parameters), snel op CPU én MPS, en grof/"beroerd" genoeg voor een kunstproject. Wissel simpel van model via het ⚙-tandwiel (`model_id`) of in `config.json`:

| Model id | Download | Opmerking |
|---|---|---|
| `segmind/tiny-sd` | ~1–2 GB | **standaard** — klein, snel, ruw |
| `hf-internal-testing/tiny-stable-diffusion-torch` | **8,8 MB** | random weights → puur ruis/"beroerd"; ideaal om razendsnel te testen op elke machine |
| `runwayml/stable-diffusion-v1-5` | ~2 GB (fp16) / ~4 GB (fp32) | de klassieker; 4–6 GB VRAM of CPU met weinig stappen |
| `stabilityai/sdxl-turbo` | ~6–7 GB (fp16) | 1–4 stappen, `guidance_scale: 0`, 8 GB VRAM |

Tip voor CPU-only: hou `steps` laag (10–20) en resolutie 512×512.

## De interface

- **Prompt** — vrije tekst.
- **Breedte/Hoogte** — resolutie, wordt automatisch afgerond op een veelvoud van 8 (64–2048).
- **Seed** — `-1` = willekeurig (wordt gelogd bij het resultaat).
- **Stappen** — aantal diffusie-stappen.
- **⚙ tandwiel** — instellingen: `system_prompt`, `negative_prompt`, `model_id`, `device` (auto/cpu/cuda/mps), stappen, guidance, seed, output-map. Wordt opgeslagen in `config.json`.

**System prompt:** Stable Diffusion kent geen echte "system prompt"; de node zet hem daarom vóór je prompt (komma-gescheiden). In het tandwiel pas je hem aan. Tip: Engels werkt beter (de CLIP-encoder is Engelstalig getraind) en houd hem kort — CLIP kapt na 77 tokens. Standaard (voor jouw stijl):

```
raw casual smartphone snapshot, ungraded unedited photo, natural available light, amateur off-center framing, awkward unusual angle, cluttered difficult composition, candid everyday average people in plain clothes, mundane setting, film grain, muted flat colors, no retouching, lo-fi
```

Output staat als PNG in `outputs/` (instelbaar).

## Inbouwen in je node-app

De node is een gewone Python-class, los van de GUI:

```python
from abstract_image_node import AbstractImageNode

node = AbstractImageNode.from_config("config.json")   # leest ⚙-instellingen
result = node.generate(
    "abstracte paarse inktvlekken",
    width=768, height=512,      # wordt automatisch veelvoud van 8
    seed=42,                    # -1 = willekeurig
    progress_cb=lambda done, total: print(f"{done}/{total}"),
)
result.image        # PIL Image (klaar om in je tegel te tonen)
result.path         # opgeslagen PNG
result.seed         # gebruikte seed
```

Meerdere tegels één model laten delen ( scheelt geheugen):

```python
node_a = AbstractImageNode.from_config("config.json")
node_b = AbstractImageNode.from_config("config.json", pipeline=node_a.load())
```

Of injecteer je eigen pipeline (handig voor tests in je app):

```python
node = AbstractImageNode(pipeline_factory=lambda settings: mijn_eigen_pipeline)
```

## Testen (zonder GPU, zonder model-download)

```bash
python3 -m unittest discover -s tests -v
```

De tests gebruiken een fake pipeline, dus ze draaien op elke machine (ook CI/Linux zonder GPU).

## Bestandsstructuur

```
abstract-image-node/
├── abstract_image_node.py   # de node zelf (inbouwbaar, geen GUI-afhankelijkheden)
├── app.py                   # losse test-app (tkinter): prompt, resolutie, ⚙-tandwiel
├── cli.py                   # headless CLI (Linux zonder scherm)
├── config.json              # instellingen (incl. system prompt) — door ⚙ te wijzigen
├── requirements.txt
└── tests/test_node.py       # tests zonder model
```
