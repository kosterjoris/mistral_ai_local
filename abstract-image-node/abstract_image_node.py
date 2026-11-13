from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional

MIN_SIZE = 64
MAX_SIZE = 2048
SIZE_STEP = 8
SEED_RANDOM = -1
MAX_SEED = 2**31 - 1


@dataclass
class NodeSettings:
    model_id: str = "segmind/tiny-sd"
    system_prompt: str = "raw casual smartphone snapshot, ungraded unedited photo, natural available light, amateur off-center framing, awkward unusual angle, cluttered difficult composition, candid everyday average people in plain clothes, mundane setting, film grain, muted flat colors, no retouching, lo-fi"
    negative_prompt: str = "text, watermark, logo, oversaturated, studio lighting, professional photoshoot, glossy retouched skin, perfect centered composition, blurry, low quality, illustration, painting, 3d render"
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = SEED_RANDOM
    device: str = "auto"
    output_dir: str = "outputs"

    @classmethod
    def from_dict(cls, data: dict) -> "NodeSettings":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def load(cls, path) -> "NodeSettings":
        path = Path(path)
        if path.exists():
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


@dataclass
class NodeResult:
    image: Any
    path: Path
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    model_id: str
    elapsed: float


class AbstractImageNode:
    """Eén tegel/node: genereert een abstract plaatje met een klein lokaal AI-model.

    In te bouwen in een node-based app: maak één instance (of deel een pipeline
    via `pipeline=`) en call `generate(...)`. GUI-onafhankelijk.
    """

    def __init__(self, settings=None, *, pipeline=None, pipeline_factory=None, config_path=None):
        self.settings = settings or NodeSettings()
        self._pipeline = pipeline
        self._pipeline_factory = pipeline_factory
        self._config_path = Path(config_path) if config_path else None
        self._lock = threading.RLock()
        self._device: Optional[str] = None

    @classmethod
    def from_config(cls, path) -> "AbstractImageNode":
        path = Path(path)
        return cls(NodeSettings.load(path), config_path=path)

    def load(self):
        with self._lock:
            if self._pipeline is None:
                self._device = self._resolve_device()
                if self._pipeline_factory is not None:
                    self._pipeline = self._pipeline_factory(self.settings)
                else:
                    self._pipeline = self._load_diffusers_pipeline()
            return self._pipeline

    def _resolve_device(self) -> str:
        requested = self.settings.device
        if requested and requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _diffusers_supports_dtype() -> bool:
        import diffusers

        try:
            major, minor = (int(part) for part in diffusers.__version__.split(".")[:2])
        except Exception:
            return False
        return (major, minor) >= (0, 32)

    def _load_diffusers_pipeline(self):
        import torch
        from diffusers import StableDiffusionPipeline

        device = self._device or self._resolve_device()
        use_fp16 = device in ("cuda", "mps")
        dtype = torch.float16 if use_fp16 else torch.float32
        model_id = self.settings.model_id
        dtype_key = "dtype" if self._diffusers_supports_dtype() else "torch_dtype"
        attempts = [
            {dtype_key: dtype, "variant": "fp16", "use_safetensors": True} if use_fp16
            else {dtype_key: dtype, "use_safetensors": True},
            {dtype_key: dtype},
        ]
        pipe = None
        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
                break
            except Exception as exc:
                last_error = exc
        if pipe is None:
            raise RuntimeError(f"Kon model '{model_id}' niet laden: {last_error}")
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def _make_generator(self, seed: int):
        try:
            import torch
        except ImportError:
            return None
        for device in (self._device or "cpu", "cpu"):
            try:
                return torch.Generator(device=device).manual_seed(seed)
            except Exception:
                continue
        return None

    @staticmethod
    def _snap_size(value) -> int:
        value = max(MIN_SIZE, min(MAX_SIZE, int(value)))
        return value - (value % SIZE_STEP)

    @staticmethod
    def build_prompt(prompt: str, system_prompt: str = "") -> str:
        parts = [p.strip().strip(",").strip() for p in (system_prompt, prompt)]
        return ", ".join(p for p in parts if p)

    def _output_dir(self) -> Path:
        out = Path(self.settings.output_dir)
        if not out.is_absolute():
            base = self._config_path.parent if self._config_path else Path.cwd()
            out = base / out
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save(self, image, seed: int, width: int, height: int) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self._output_dir() / f"abstract_{stamp}_{seed}_{width}x{height}.png"
        image.save(str(path))
        return path

    def _step_callback(self, progress_cb: Optional[Callable[[int, int], None]], steps: int):
        if progress_cb is None:
            return None

        def callback(pipe, step, timestep, callback_kwargs):
            progress_cb(step + 1, steps)
            return callback_kwargs

        return callback

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        system_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> NodeResult:
        """Genereer één plaatje. Resolutie wordt automatisch op een veelvoud van 8 afgerond."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt mag niet leeg zijn.")
        s = self.settings
        width = self._snap_size(width)
        height = self._snap_size(height)
        system_prompt = s.system_prompt if system_prompt is None else system_prompt
        negative_prompt = s.negative_prompt if negative_prompt is None else negative_prompt
        steps = int(s.steps if steps is None else steps)
        guidance_scale = float(s.guidance_scale if guidance_scale is None else guidance_scale)
        seed = int(s.seed if seed is None else seed)
        if seed == SEED_RANDOM or seed < 0:
            seed = random.SystemRandom().randrange(0, MAX_SEED)
        full_prompt = self.build_prompt(prompt, system_prompt)
        started = time.perf_counter()
        with self._lock:
            pipe = self.load()
            callback = self._step_callback(progress_cb, steps)
            call_kwargs = dict(
                prompt=full_prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=self._make_generator(seed),
            )
            if callback is not None:
                try:
                    output = pipe(**call_kwargs, callback_on_step_end=callback)
                except TypeError:
                    output = pipe(**call_kwargs)
            else:
                output = pipe(**call_kwargs)
            image = output.images[0]
        path = self._save(image, seed, width, height)
        return NodeResult(
            image=image,
            path=path,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            model_id=s.model_id,
            elapsed=time.perf_counter() - started,
        )

    def update_settings(self, **kwargs) -> "AbstractImageNode":
        valid = {f.name for f in fields(NodeSettings)}
        unknown = set(kwargs) - valid
        if unknown:
            raise ValueError(f"Onbekende instelling(en): {', '.join(sorted(unknown))}")
        reload_needed = any(k in ("model_id", "device") for k in kwargs)
        for key, value in kwargs.items():
            setattr(self.settings, key, value)
        if reload_needed:
            with self._lock:
                self._pipeline = None
                self._device = None
        return self

    def save_settings(self, path=None) -> Path:
        path = Path(path) if path else (self._config_path or Path("config.json"))
        return self.settings.save(path)
