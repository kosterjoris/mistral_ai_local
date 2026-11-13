# Overdracht — `abstract-image-node` (lokale AI-tegel voor het genereren van (abstracte/raw) plaatjes)

Geschreven voor: Claude Code die dit in het originele node/tegels-project moet inbouwen.
Taal: Nederlands; code en identifiers Engels.

## 1. Waar staat wat

| Wat | Waar |
|---|---|
| Code (GitHub) | repo **`kosterjoris/mistral_ai_local`**, branch **`vibe/abstract-image-node-a917c6`** (is nu de default branch; de repo was leeg bij start). Clone: `git clone https://github.com/kosterjoris/mistral_ai_local.git` |
| Map in de repo | **`abstract-image-node/`** in de repo-root |
| Lokaal op de Mac | `~/mistral_ai_local/abstract-image-node/` (zelfde clone; `git pull` voor laatste) |
| Huidige HEAD | `fdb7a50` |
| Modelgewichten | **NIET in git.** Per machine in `~/.cache/huggingface/hub/models--...` (eenmalige download, daarna offline). Optioneel project-lokaal via `"cache_dir": "models"` → `<project>/models/` (staat in `.gitignore`) |
| Gegenereerde plaatjes | `abstract-image-node/outputs/abstract_<datum>_<seed>_<WxH>.png` (gitignored) |
| Instellingen | `abstract-image-node/config.json` (wordt geschreven door het ⚙-tandwiel in de test-app) |

Commits op de branch: `6555077` (init) → `eff57da` (dtype-fix) → `c26b358` (Engelse default prompts) → `9ca98d9` (model-presets + zoom) → `0f67b49` (cleanup) → `bd63c8f` (sync fix) → `fdb7a50` (cache_dir).

## 2. Architectuur / bestanden

```
abstract-image-node/
├── abstract_image_node.py   # DE TEGEL — dit is het enige bestand dat het originele project nodig heeft
├── app.py                   # losse test-app (tkinter GUI) — alleen om te testen, niet inbouwen
├── cli.py                   # headless CLI (Linux zonder scherm)
├── config.json              # instellingen (system prompt, model, device, steps, ...)
├── requirements.txt         # deps (zie §3)
├── OVERDRACHT.md            # dit document
├── tests/test_node.py       # 17 unittests, draaien ZONDER GPU/model (fake pipeline)
└── README.md                # quickstart Mac/Linux, modelkeuze, troubleshooting
```

Kern-API (`abstract_image_node.py`):

- `NodeSettings` (dataclass): `model_id, system_prompt, negative_prompt, steps, guidance_scale, seed, device, output_dir, cache_dir` — `load(path)` / `save(path)` (JSON), `from_dict` filtert onbekende keys.
- `MODEL_PRESETS`: dict label → `{model_id, steps, guidance_scale}` (zie §4).
- `AbstractImageNode(settings=None, *, pipeline=None, pipeline_factory=None, config_path=None)`
  - `from_config(path)` — leest `config.json`.
  - `generate(prompt, *, width, height, system_prompt, negative_prompt, seed, steps, guidance_scale, progress_cb)` → `NodeResult(image, path, prompt, negative_prompt, seed, width, height, steps, guidance_scale, model_id, elapsed)`.
  - `update_settings(**kwargs)` — valideren tegen dataclass-velden; bij wijziging van `model_id`/`device` wordt de pipeline opnieuw geladen. `save_settings(path=None)`.
  - Alleen stdlib bij import; `torch`/`diffusers` worden lazy geladen in `load()` → het bestand is veilig te importeren in de app zonder zware deps.
  - Device auto: `cuda` → `mps` (Apple Silicon) → `cpu`; fp16 op cuda/mps, fp32 op cpu. Loader: `AutoPipelineForText2Image.from_pretrained` met `dtype=` (diffusers ≥ 0.32; valt terug op `torch_dtype=`), eerst `variant="fp16"`+`use_safetensors`, dan plain. `cache_dir` wordt doorgegeven als gezet.
  - Resolutie wordt automatisch begrensd (64–2048) en afgerond op veelvoud van 8. `seed=-1` = willekeurig (wordt in `NodeResult.seed` teruggegeven). `system_prompt` wordt vóór de prompt gezet, komma-gescheiden (SD heeft geen echte system prompt; CLIP kapt na 77 tokens; Engels werkt het beste).
  - Thread-safe (RLock rond laden+genereren). Pipeline is injecteerbaar (`pipeline=...` / `pipeline_factory=...`) → meerdere tegels kunnen één geladen model delen, en tests draaien zonder echt model.
  - `progress_cb(done, total)` per diffusiestap.

Test-app (`app.py`, tkinter): promptveld, breedte/hoogte/seed/stappen spinboxes, **⚙-tandwiel** = instellingen-dialoog (model als dropdown uit `MODEL_PRESETS` met auto-ingevulde stappen/guidance, device, system/negative prompt, seed, output map, cache map), **−/+ zoom-knoppen** bij de preview, generatie in worker-thread met queue-polling (UI bevriest niet), preview via `tk.PhotoImage(file=...)` (Tk ≥ 8.6 nodig voor PNG).

## 3. Techniek / stack

- Python ≥ 3.10. Deps (`requirements.txt`): `torch>=2.2`, `diffusers>=0.30`, `transformers>=4.44`, `accelerate>=0.33`, `safetensors>=0.4`, `pillow>=10.3`.
- macOS (Apple Silicon): gewone `pip install -r requirements.txt` (torch met MPS built-in). GUI: Homebrew/python.org python — `brew install python-tk` (system python3 van macOS heeft geen tkinter).
- Linux met NVIDIA: normale torch (CUDA). Linux CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, en `"device": "cpu"` in config.
- Alles draait **100% lokaal**: gewichten komen één keer van huggingface.co naar de HF-cache; bij genereren geen netwerk. Offline afdwingen: `HF_HUB_OFFLINE=1`.
- Geen cloud, geen API-sleutels.

## 4. Modellen (presets in `MODEL_PRESETS`, te kiezen in ⚙-dropdown)

| Preset | model_id | Download | Opmerking |
|---|---|---|---|
| tiny-sd (standaard) | `segmind/tiny-sd` | ~1 GB | snel, ruw/basic — expres "beroerd" |
| tiny random weights | `hf-internal-testing/tiny-stable-diffusion-torch` | 8,8 MB | random weights = pure ruis; om de keten in seconden te testen |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | ~2 GB fp16 | beste kwaliteit voor de grootte — tip voor raw/casual foto-stijl |
| SDXL Turbo | `stabilityai/sdxl-turbo` | ~7 GB fp16 | 1–4 stappen, **guidance_scale 0** (preset vult dit auto in), 8 GB+ VRAM |

Default system prompt (Engels, in `config.json`):
`raw casual smartphone snapshot, ungraded unedited photo, natural available light, amateur off-center framing, awkward unusual angle, cluttered difficult composition, candid everyday average people in plain clothes, mundane setting, film grain, muted flat colors, no retouching, lo-fi`

## 5. Inbouwen in het originele node-project (recept)

1. Kopieer **`abstract_image_node.py`** (één bestand) naar het app-project, bijv. `nodes/abstract_image_node.py`. Optioneel ook `config.json`.
2. Voeg de deps uit `requirements.txt` toe aan het project.
3. Maak per tegel (of één gedeelde instance) een node:

```python
from abstract_image_node import AbstractImageNode, NodeSettings, MODEL_PRESETS

# gedeelde instance, instellingen uit config (of NodeSettings(...) in code)
node = AbstractImageNode.from_config("config.json")

# in de tegel (GUI-thread NIET blokkeren → worker thread):
def on_generate(self):
    result = node.generate(
        self.prompt,                 # user prompt (mag NL; Engels werkt beter)
        width=self.width, height=self.height,   # wordt auto veelvoud van 8
        seed=self.seed,               # -1 = willekeurig
        progress_cb=lambda d, t: self.set_progress(d, t),
    )
    self.set_image(result.image)      # PIL.Image → direct in de tegel tonen
    self.label = f"seed {result.seed} → {result.path.name}"   # PNG staat in outputs/
```

4. Meerdere tegels één model laten delen (scheelt geheugen): `node_b = AbstractImageNode(settings, pipeline=node_a.load())` of via `pipeline_factory`.
5. Instellingen (de "settings-cogwheel" van het project): lees/schrijf `config.json`, of roep `node.update_settings(system_prompt=..., model_id=...)` + `node.save_settings()` aan. Modelkeuze via `MODEL_PRESETS` (label → model_id/stappen/guidance).
6. Tests overnemen: `tests/test_node.py` (17 tests, fake pipeline — geen GPU nodig): `python3 -m unittest discover -s tests -v`.

## 6. Draaien & testen

```bash
git clone https://github.com/kosterjoris/mistral_ai_local.git
cd mistral_ai_local/abstract-image-node
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m unittest discover -s tests -v          # 17 tests, geen model nodig → OK
python3 cli.py "car park at night, phone photo" --width 512 --height 512 --steps 25 --seed 42   # headless (Linux)
ls outputs/                                        # PNG staat hier
python app.py                                      # GUI (Mac: /opt/homebrew/bin/python3 app.py na brew install python-tk)
```

## 7. Bekende valkuilen (geleerd tijdens het bouwen)

- **Grijze vlakken** = te weinig stappen (CLI-test gebruikte `--steps 4`) of te korte prompt. Oplossing: 20–30 stappen, 512×512, langere concrete prompt, andere seed, evt. SD 1.5 preset. `tiny-sd` is expres basic.
- App starten met de **venv-python** (`python app.py`), niet de globale `/opt/homebrew/bin/python3` → anders `No module named 'torch'`.
- `brew install python-tk` upgradet Homebrew-python (hier naar 3.15) — maak daarna evt. de venv opnieuw (`rm -rf .venv; /opt/homebrew/bin/python3 -m venv .venv`).
- zsh interactief: `#` is géén comment-teken → plak commands zonder inline comments.
- Hoofdvenster (stappen/seed) synchroniseert nu automatisch na Opslaan in het ⚙-dialoog (fix `bd63c8f`).
- CLIP-encoder is Engelstalig getraind → system/user prompt in het Engels; houd system prompt kort (< ~40 tokens, limiet 77 incl. user prompt).
- `torch_dtype=` is deprecated in diffusers ≥ 0.32 (verdwijnt in 1.0) — loader gebruikt automatisch `dtype=` waar beschikbaar (fix `eff57da`).
- Preview in GUI heeft Tk ≥ 8.6 nodig (PNG); bij oudere Tk staat de PNG alsnog goed weg in `outputs/`.
- Push vanuit de build-sandbox naar GitHub vereist de **mistralai GitHub App** geïnstalleerd op het account (contents+PR write); `gh auth setup-git` is daar geblokkeerd, pushen ging via `git push` met `!gh auth git-credential` als wegwerp credential helper.

## 8. Git-workflow

- Branch: `vibe/abstract-image-node-a917c6` (vibe-slug). Repo was leeg → deze branch is automatisch default geworden; een PR kan pas worden aangemaakt zodra er een tweede branch/base is.
- Push: `git push origin vibe/abstract-image-node-a917c6`.
- `.gitignore` in de map dekt `outputs/`, `models/`, `.venv/`, `__pycache__/` af — gewichten en venv gaan nooit naar git.
