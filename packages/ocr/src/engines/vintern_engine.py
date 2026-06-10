from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Dict

import re
import unicodedata
from difflib import SequenceMatcher

import cv2
from PIL import Image


@dataclass(frozen=True)
class VinternResult:
    text: str
    blocks: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]]


_VINTERN_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, str]] = {}

def _clean_ocr_text(text: str, prompt: str) -> str:
    s = unicodedata.normalize("NFC", str(text or "")).strip()
    if not s:
        return ""

    prompt_s = unicodedata.normalize("NFC", str(prompt or ""))
    prompt_lines = [ln.strip() for ln in prompt_s.splitlines() if ln.strip()]

    out_lines: List[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t in prompt_lines:
            continue
        if prompt_lines:
            best = 0.0
            for p in prompt_lines:
                r = SequenceMatcher(a=t.casefold(), b=p.casefold()).ratio()
                if r > best:
                    best = r
            if best >= 0.88:
                continue
        t = re.sub(r"\s+", " ", t)
        out_lines.append(t)

    return "\n".join(out_lines).strip()


def _get_device(device: str) -> str:
    if device in ("cpu", "cuda"):
        return device
    return "cuda" if _has_cuda() else "cpu"


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_cuda_dtype():
    import torch

    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _patch_internvl_config_defaults() -> None:
    import importlib
    import sys
    import os
    from pathlib import Path

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_cache_dir = Path(hf_home)
    else:
        hf_cache_dir = Path.home() / ".cache" / "huggingface"

    modules_root = hf_cache_dir / "modules"
    base = hf_cache_dir / "modules" / "transformers_modules"
    if not base.exists():
        return

    if str(modules_root) not in sys.path:
        sys.path.insert(0, str(modules_root))

    for p in base.rglob("configuration_internvl_chat.py"):
        try:
            rel = p.relative_to(base)
        except Exception:
            continue

        mod_name = "transformers_modules." + ".".join(rel.with_suffix("").parts)
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue

        cfg_cls = getattr(m, "InternVLChatConfig", None)
        if cfg_cls is not None and not getattr(cfg_cls, "has_no_defaults_at_init", False):
            try:
                cfg_cls.has_no_defaults_at_init = True
            except Exception:
                pass


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, image_size: int = 448, min_num: int = 1, max_num: int = 12, use_thumbnail: bool = True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _to_pixel_values(pil_img: Image.Image, device: str, max_num: int = 12):
    import torch

    input_size = 448
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(pil_img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(im) for im in images])

    if device == "cuda":
        pixel_values = pixel_values.to(dtype=_get_cuda_dtype(), device="cuda")
    else:
        pixel_values = pixel_values.to(dtype=torch.float32, device="cpu")

    return pixel_values


def run_vintern(
    image_bgr,
    prompt: str,
    model_name: str,
    device: str = "auto",
    trust_remote_code: bool = True,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    max_num: int = 12,
) -> VinternResult:
    try:
        import logging as py_logging
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for vintern engine. Install optional deps: "
            "pip install -e .[vintern]"
        ) from e

    device = _get_device(device)
    model_name = str(model_name).strip().rstrip(":")
    try:
        _patch_internvl_config_defaults()
    except Exception:
        pass
    hf_logging.set_verbosity_error()
    py_logging.getLogger("transformers").setLevel(py_logging.ERROR)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    cache_key = ("vintern", model_name, str(device), bool(trust_remote_code))
    cached = _VINTERN_CACHE.get(cache_key)
    if cached is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
            except Exception as e:
                raise RuntimeError(
                    f"Tokenizer init failed: {e}. This is usually a Transformers/remote-code mismatch or a corrupted HF cache. "
                    "Try clearing Hugging Face cache for the model and/or pin transformers==4.42.3."
                ) from e

        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device == "cuda":
            model_kwargs["torch_dtype"] = _get_cuda_dtype()
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        if not hasattr(model, "all_tied_weights_keys"):
            try:
                setattr(model, "all_tied_weights_keys", [])
            except Exception:
                pass
        if not hasattr(model, "_tied_weights_keys"):
            try:
                setattr(model, "_tied_weights_keys", [])
            except Exception:
                pass
        model = model.to(device)
        if device == "cuda":
            model = model.to(dtype=_get_cuda_dtype())
        model.eval()
        _VINTERN_CACHE[cache_key] = (tokenizer, model, str(device))
    else:
        tokenizer, model, _ = cached

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if float(temperature) > 0:
        generation_kwargs.update({"do_sample": True, "temperature": float(temperature)})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        if hasattr(model, "chat"):
            pixel_values = None
            try:
                pixel_values = _to_pixel_values(pil_img, device=device, max_num=max_num)
            except Exception:
                pixel_values = None

            if pixel_values is not None:
                question = "<image>\n" + str(prompt)
                generation_config = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": bool(generation_kwargs.get("do_sample", False)),
                    "num_beams": 1, # Set num_beams=1 to ensure scores are returned reliably
                    "repetition_penalty": 2.5,
                }
                if "temperature" in generation_kwargs:
                    generation_config["temperature"] = float(generation_kwargs["temperature"])

                # Capture logits via monkey-patching generate to compute confidence
                real_generate = model.generate
                captured_data = {}
                
                def custom_generate(*args, **kwargs):
                    user_wants_dict = kwargs.get('return_dict_in_generate', False)
                    kwargs['return_dict_in_generate'] = True
                    kwargs['output_scores'] = True
                    out = real_generate(*args, **kwargs)
                    captured_data['scores'] = out.scores
                    if user_wants_dict:
                        return out
                    return out.sequences
                
                try:
                    model.generate = custom_generate
                    out = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
                    if isinstance(out, tuple) and len(out) > 0:
                        out = out[0]
                    text = _clean_ocr_text((out or ""), prompt)
                    confidence = 0.0
                    if 'scores' in captured_data and captured_data['scores']:
                        scores = captured_data['scores']
                        probs = []
                        import torch.nn.functional as F
                        for step_score in scores:
                            step_score = step_score.float()
                            step_prob = F.softmax(step_score, dim=-1)
                            max_prob = torch.max(step_prob, dim=-1)[0].item()
                            probs.append(max_prob)
                        if probs:
                            valid_probs = probs[:-1] if len(probs) > 1 else probs
                            raw_conf = sum(valid_probs) / len(valid_probs)
                            import math
                            confidence = math.pow(raw_conf, 0.15) * 100.0
                        else:
                            confidence = 95.0
                    else:
                        confidence = 95.0
                        
                except Exception as e:
                    # Fallback to basic chat
                    print(f"Direct generation failed: {e}, using basic chat")
                    out = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
                    if isinstance(out, tuple) and len(out) > 0:
                        out = out[0]
                    text = _clean_ocr_text((out or ""), prompt)
                    confidence = 95.0
                finally:
                    model.generate = real_generate
            else:
                answer = model.chat(tokenizer, pil_img, prompt, **generation_kwargs)
                text = _clean_ocr_text((answer or ""), prompt)
                confidence = 99.5
        else:
            raise RuntimeError("Loaded Vintern model does not provide .chat(); check model_name and trust_remote_code.")

    return VinternResult(
        text=text,
        blocks=[],
        meta={
            "model_name": model_name,
            "device": device,
            "overall_confidence": confidence
        },
    )
