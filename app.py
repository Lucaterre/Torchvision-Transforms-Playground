#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torchvision Transforms Playground (Gradio framework)

Interactive sandbox to transforming images using torchvision that includes this features:
- Upload one or multiple images
- Toggle transforms and tune parameters
- Preview one example per enabled transform + a final MIX pipeline with multiple random variants
- See a dynamically generated torchvision Compose code snippet
- Switch UI language (EN/FR)
- Disable all transforms in one click
- Quick links to torchvision documentation per section

Usage:
    python3 app.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_pil_image

# Assets
DEFAULT_I18N_PATH = "assets/i18n.json"
DEFAULT_CSS_PATH = "assets/styles.css"


# Small utilities (image / html / i18n)
def load_texts_json(path: str) -> str:
    """
    Load i18n JSON string from file.

    :param path: Path to JSON file.
    :type path: str
    :return: JSON string.
    :rtype: str
    """
    return Path(path).read_text(encoding="utf-8")


def load_css(path: str) -> str:
    """
    Load CSS string from file.

    :param path: Path to CSS file.
    :type path: str
    :return: CSS string.
    :rtype: str
    """
    return Path(path).read_text(encoding="utf-8")


class I18N:
    """
    Simple i18n manager backed by a JSON string.

    :param texts_json: JSON string holding UI texts for each language.
    :type texts_json: str
    :param default_lang: Default language key (e.g. "EN").
    :type default_lang: str
    """

    def __init__(self, texts_json: str, default_lang: str = "EN") -> None:
        self._texts = json.loads(texts_json)
        self._default_lang = default_lang

    def get(self, lang: str, key: str, default: Optional[str] = None) -> str:
        """
        Get a text key for a given language.

        :param lang: Language key (e.g. "EN", "FR").
        :type lang: str
        :param key: Text key.
        :type key: str
        :param default: Default value if missing.
        :type default: Optional[str]
        :return: Text value.
        :rtype: str
        """
        return self._texts.get(lang, {}).get(
            key, default if default is not None else ""
        )

    def section(self, lang: str, section_key: str) -> str:
        """
        Get a section name (translated).

        :param lang: Language key.
        :type lang: str
        :param section_key: Section identifier (e.g. "geometric").
        :type section_key: str
        :return: Section label.
        :rtype: str
        """
        return (
            self._texts.get(lang, {}).get("sections", {}).get(section_key, section_key)
        )

    def subtitles(self, lang: str) -> List[str]:
        """
        Get the list of subtitles for a language.

        :param lang: Language key.
        :type lang: str
        :return: Subtitles list.
        :rtype: List[str]
        """
        return list(self._texts.get(lang, {}).get("app_subtitles", []))


def status_dot(active: bool) -> str:
    """
    Render a small colored dot (HTML).

    :param active: Whether the section is active.
    :type active: bool
    :return: HTML span for a dot.
    :rtype: str
    """
    color = "#22c55e" if active else "#94a3b8"  # green / gray
    return (
        "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{color};margin-right:8px;'></span>"
    )


def as_pil_list(gallery_value: Any) -> List[Image.Image]:
    """
    Convert a Gradio Gallery input value to a list of PIL images.

    :param gallery_value: Gallery value from gr.Gallery.
    :type gallery_value: Any
    :return: List of PIL.Image objects.
    :rtype: List[Image.Image]
    """
    if not gallery_value:
        return []
    imgs: List[Image.Image] = []
    for item in gallery_value:
        if isinstance(item, tuple) and len(item) >= 1:
            imgs.append(item[0])
        else:
            imgs.append(item)
    return imgs


def ensure_pil(x: Any) -> Image.Image:
    """
    Ensure output is a PIL image (convert torch.Tensor if needed).

    :param x: Transform output (PIL.Image or torch.Tensor).
    :type x: Any
    :return: PIL image.
    :rtype: PIL.Image.Image
    :raises TypeError: If unsupported type.
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        return to_pil_image(x.clamp(0, 1))
    raise TypeError(f"Unsupported output type: {type(x)}")


@dataclass
class TransformItem:
    """
    A single transform descriptor.

    :param name: Display name.
    :type name: str
    :param op: Transform object OR a special sentinel string.
    :type op: Any
    """

    name: str
    op: Any


class TransformFactory:
    """
    Factory for building:
    - a list of enabled single transforms (one-by-one examples)
    - the final Compose pipeline (MIX)

    This encapsulates transform construction and keeps UI code clean.
    """

    TENSOR_ERASE_ONLY = "TENSOR_ERASE_ONLY"
    TENSOR_NORM_ONLY = "TENSOR_NORM_ONLY"

    def build_single_transforms(self, p: Dict[str, Any]) -> List[TransformItem]:
        """
        Build a list of single transforms (one transform = one operation).

        :param p: Parameters dict (toggles + params).
        :type p: Dict[str, Any]
        :return: List of enabled transforms.
        :rtype: List[TransformItem]
        """
        L: List[TransformItem] = []

        # Geometric
        if p["use_pad"]:
            L.append(
                TransformItem(
                    "Pad",
                    T.Pad(
                        padding=int(p["pad_px"]),
                        fill=int(p["pad_fill"]),
                        padding_mode=p["pad_mode"],
                    ),
                )
            )
        if p["use_resize"]:
            L.append(
                TransformItem(
                    "Resize", T.Resize((int(p["resize_size"]), int(p["resize_size"])))
                )
            )
        if p["use_center_crop"]:
            L.append(
                TransformItem(
                    "CenterCrop",
                    T.CenterCrop((int(p["crop_size"]), int(p["crop_size"]))),
                )
            )
        if p["use_five_crop"]:
            L.append(
                TransformItem(
                    "FiveCrop",
                    T.FiveCrop((int(p["five_crop_size"]), int(p["five_crop_size"]))),
                )
            )
        if p["use_random_perspective"]:
            L.append(
                TransformItem(
                    "RandomPerspective",
                    T.RandomPerspective(
                        distortion_scale=float(p["persp_dist"]), p=float(p["persp_p"])
                    ),
                )
            )
        if p["use_random_rotation"]:
            L.append(
                TransformItem(
                    "RandomRotation", T.RandomRotation(degrees=int(p["rot_deg"]))
                )
            )
        if p["use_random_affine"]:
            L.append(
                TransformItem(
                    "RandomAffine",
                    T.RandomAffine(
                        degrees=int(p["aff_deg"]),
                        translate=(
                            float(p["aff_translate"]),
                            float(p["aff_translate"]),
                        ),
                        scale=(float(p["aff_scale_min"]), float(p["aff_scale_max"])),
                        shear=int(p["aff_shear"]),
                    ),
                )
            )
        if p["use_elastic"]:
            L.append(
                TransformItem(
                    "ElasticTransform",
                    T.ElasticTransform(
                        alpha=float(p["elastic_alpha"]), sigma=float(p["elastic_sigma"])
                    ),
                )
            )
        if p["use_random_crop"]:
            L.append(
                TransformItem(
                    "RandomCrop",
                    T.RandomCrop((int(p["rand_crop_size"]), int(p["rand_crop_size"]))),
                )
            )
        if p["use_rrc"]:
            L.append(
                TransformItem(
                    "RandomResizedCrop",
                    T.RandomResizedCrop(
                        (int(p["rrc_size"]), int(p["rrc_size"])),
                        scale=(float(p["rrc_scale_min"]), float(p["rrc_scale_max"])),
                    ),
                )
            )

        # Photometric
        if p["use_grayscale"]:
            L.append(
                TransformItem(
                    "Grayscale",
                    T.Grayscale(num_output_channels=int(p["gray_channels"])),
                )
            )
        if p["use_cj"]:
            L.append(
                TransformItem(
                    "ColorJitter",
                    T.ColorJitter(
                        brightness=float(p["cj_b"]),
                        contrast=float(p["cj_c"]),
                        saturation=float(p["cj_s"]),
                        hue=float(p["cj_h"]),
                    ),
                )
            )
        if p["use_blur"]:
            k = int(p["blur_k"])
            if k % 2 == 0:
                k += 1
            L.append(
                TransformItem(
                    "GaussianBlur",
                    T.GaussianBlur(
                        kernel_size=k,
                        sigma=(float(p["blur_sigma_min"]), float(p["blur_sigma_max"])),
                    ),
                )
            )
        if p["use_inv"]:
            L.append(TransformItem("RandomInvert", T.RandomInvert(p=float(p["inv_p"]))))
        if p["use_post"]:
            L.append(
                TransformItem(
                    "RandomPosterize",
                    T.RandomPosterize(bits=int(p["post_bits"]), p=float(p["post_p"])),
                )
            )
        if p["use_sol"]:
            L.append(
                TransformItem(
                    "RandomSolarize",
                    T.RandomSolarize(
                        threshold=int(p["sol_thresh"]), p=float(p["sol_p"])
                    ),
                )
            )
        if p["use_sharp"]:
            L.append(
                TransformItem(
                    "RandomAdjustSharpness",
                    T.RandomAdjustSharpness(
                        sharpness_factor=float(p["sharp_factor"]), p=float(p["sharp_p"])
                    ),
                )
            )
        if p["use_autoc"]:
            L.append(TransformItem("RandomAutocontrast", T.RandomAutocontrast()))
        if p["use_eq"]:
            L.append(TransformItem("RandomEqualize", T.RandomEqualize()))
        if p["use_jpeg"]:
            L.append(
                TransformItem(
                    "JPEG", T.JPEG(quality=(int(p["jpeg_qmin"]), int(p["jpeg_qmax"])))
                )
            )

        # Policies
        if p["use_autoaugment"]:
            policy = getattr(T.AutoAugmentPolicy, p["aa_policy"])
            L.append(TransformItem("AutoAugment", T.AutoAugment(policy=policy)))
        if p["use_randaugment"]:
            L.append(
                TransformItem(
                    "RandAugment",
                    T.RandAugment(
                        num_ops=int(p["ra_num_ops"]), magnitude=int(p["ra_mag"])
                    ),
                )
            )
        if p["use_trivial"]:
            L.append(
                TransformItem(
                    "TrivialAugmentWide",
                    T.TrivialAugmentWide(num_magnitude_bins=int(p["tw_bins"])),
                )
            )
        if p["use_augmix"]:
            L.append(
                TransformItem(
                    "AugMix",
                    T.AugMix(
                        severity=int(p["am_severity"]),
                        mixture_width=int(p["am_width"]),
                        chain_depth=int(p["am_depth"]),
                        alpha=float(p["am_alpha"]),
                    ),
                )
            )

        # Randomly-applied
        if p["use_hflip"]:
            L.append(
                TransformItem(
                    "RandomHorizontalFlip",
                    T.RandomHorizontalFlip(p=float(p["hflip_p"])),
                )
            )
        if p["use_vflip"]:
            L.append(
                TransformItem(
                    "RandomVerticalFlip", T.RandomVerticalFlip(p=float(p["vflip_p"]))
                )
            )
        if p["use_random_apply"]:
            inner = [T.RandomCrop((int(p["ra_crop"]), int(p["ra_crop"])))]
            L.append(
                TransformItem(
                    "RandomApply(RandomCrop)",
                    T.RandomApply(transforms=inner, p=float(p["ra_p"])),
                )
            )

        # Tensor bonus as single examples
        if p["use_erase"]:
            L.append(TransformItem("RandomErasing (tensor)", self.TENSOR_ERASE_ONLY))
        if p["use_norm"]:
            L.append(TransformItem("Normalize (tensor)", self.TENSOR_NORM_ONLY))

        return L

    def build_compose(self, p: Dict[str, Any]) -> T.Compose:
        """
        Build the final Compose pipeline (MIX).

        :param p: Parameters dict.
        :type p: Dict[str, Any]
        :return: Torchvision v2 Compose transform.
        :rtype: torchvision.transforms.v2.Compose
        """
        transforms: List[Any] = []

        # Geometric
        if p["use_pad"]:
            transforms.append(
                T.Pad(
                    padding=int(p["pad_px"]),
                    fill=int(p["pad_fill"]),
                    padding_mode=p["pad_mode"],
                )
            )
        if p["use_resize"]:
            transforms.append(T.Resize((int(p["resize_size"]), int(p["resize_size"]))))
        if p["use_center_crop"]:
            transforms.append(T.CenterCrop((int(p["crop_size"]), int(p["crop_size"]))))
        if p["use_five_crop"]:
            transforms.append(
                T.FiveCrop((int(p["five_crop_size"]), int(p["five_crop_size"])))
            )  # returns 5
        if p["use_random_perspective"]:
            transforms.append(
                T.RandomPerspective(
                    distortion_scale=float(p["persp_dist"]), p=float(p["persp_p"])
                )
            )
        if p["use_random_rotation"]:
            transforms.append(T.RandomRotation(degrees=int(p["rot_deg"])))
        if p["use_random_affine"]:
            transforms.append(
                T.RandomAffine(
                    degrees=int(p["aff_deg"]),
                    translate=(float(p["aff_translate"]), float(p["aff_translate"])),
                    scale=(float(p["aff_scale_min"]), float(p["aff_scale_max"])),
                    shear=int(p["aff_shear"]),
                )
            )
        if p["use_elastic"]:
            transforms.append(
                T.ElasticTransform(
                    alpha=float(p["elastic_alpha"]), sigma=float(p["elastic_sigma"])
                )
            )
        if p["use_random_crop"]:
            transforms.append(
                T.RandomCrop((int(p["rand_crop_size"]), int(p["rand_crop_size"])))
            )
        if p["use_rrc"]:
            transforms.append(
                T.RandomResizedCrop(
                    (int(p["rrc_size"]), int(p["rrc_size"])),
                    scale=(float(p["rrc_scale_min"]), float(p["rrc_scale_max"])),
                )
            )

        # Photometric
        if p["use_grayscale"]:
            transforms.append(T.Grayscale(num_output_channels=int(p["gray_channels"])))
        if p["use_cj"]:
            transforms.append(
                T.ColorJitter(
                    brightness=float(p["cj_b"]),
                    contrast=float(p["cj_c"]),
                    saturation=float(p["cj_s"]),
                    hue=float(p["cj_h"]),
                )
            )
        if p["use_blur"]:
            k = int(p["blur_k"])
            if k % 2 == 0:
                k += 1
            transforms.append(
                T.GaussianBlur(
                    kernel_size=k,
                    sigma=(float(p["blur_sigma_min"]), float(p["blur_sigma_max"])),
                )
            )
        if p["use_inv"]:
            transforms.append(T.RandomInvert(p=float(p["inv_p"])))
        if p["use_post"]:
            transforms.append(
                T.RandomPosterize(bits=int(p["post_bits"]), p=float(p["post_p"]))
            )
        if p["use_sol"]:
            transforms.append(
                T.RandomSolarize(threshold=int(p["sol_thresh"]), p=float(p["sol_p"]))
            )
        if p["use_sharp"]:
            transforms.append(
                T.RandomAdjustSharpness(
                    sharpness_factor=float(p["sharp_factor"]), p=float(p["sharp_p"])
                )
            )
        if p["use_autoc"]:
            transforms.append(T.RandomAutocontrast())
        if p["use_eq"]:
            transforms.append(T.RandomEqualize())
        if p["use_jpeg"]:
            transforms.append(
                T.JPEG(quality=(int(p["jpeg_qmin"]), int(p["jpeg_qmax"])))
            )

        # Policies
        if p["use_autoaugment"]:
            policy = getattr(T.AutoAugmentPolicy, p["aa_policy"])
            transforms.append(T.AutoAugment(policy=policy))
        if p["use_randaugment"]:
            transforms.append(
                T.RandAugment(num_ops=int(p["ra_num_ops"]), magnitude=int(p["ra_mag"]))
            )
        if p["use_trivial"]:
            transforms.append(
                T.TrivialAugmentWide(num_magnitude_bins=int(p["tw_bins"]))
            )
        if p["use_augmix"]:
            transforms.append(
                T.AugMix(
                    severity=int(p["am_severity"]),
                    mixture_width=int(p["am_width"]),
                    chain_depth=int(p["am_depth"]),
                    alpha=float(p["am_alpha"]),
                )
            )

        # Randomly-applied
        if p["use_hflip"]:
            transforms.append(T.RandomHorizontalFlip(p=float(p["hflip_p"])))
        if p["use_vflip"]:
            transforms.append(T.RandomVerticalFlip(p=float(p["vflip_p"])))
        if p["use_random_apply"]:
            inner = [T.RandomCrop((int(p["ra_crop"]), int(p["ra_crop"])))]
            transforms.append(T.RandomApply(transforms=inner, p=float(p["ra_p"])))

        # Tensor-only
        need_tensor = p["use_erase"] or p["use_norm"]
        if need_tensor:
            transforms.append(T.ToImage())
            transforms.append(T.ToDtype(torch.float32, scale=True))

            if p["use_erase"]:
                transforms.append(
                    T.RandomErasing(
                        p=float(p["erase_p"]),
                        scale=(
                            float(p["erase_scale_min"]),
                            float(p["erase_scale_max"]),
                        ),
                        ratio=(
                            float(p["erase_ratio_min"]),
                            float(p["erase_ratio_max"]),
                        ),
                        value="random",
                    )
                )

            if p["use_norm"]:
                mean = [float(x.strip()) for x in str(p["norm_mean"]).split(",")]
                std = [float(x.strip()) for x in str(p["norm_std"]).split(",")]
                transforms.append(T.Normalize(mean=mean, std=std))

        return T.Compose(transforms)

    def tensor_only_example(self, p: Dict[str, Any], which: str) -> T.Compose:
        """
        Create a local tensor-only pipeline used for single-transform previews.

        :param p: Parameters dict.
        :type p: Dict[str, Any]
        :param which: Sentinel ("TENSOR_ERASE_ONLY" or "TENSOR_NORM_ONLY").
        :type which: str
        :return: Compose pipeline that converts image to tensor then applies the tensor op.
        :rtype: torchvision.transforms.v2.Compose
        """
        base = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]

        if which == self.TENSOR_ERASE_ONLY:
            base.append(
                T.RandomErasing(
                    p=float(p["erase_p"]),
                    scale=(float(p["erase_scale_min"]), float(p["erase_scale_max"])),
                    ratio=(float(p["erase_ratio_min"]), float(p["erase_ratio_max"])),
                    value="random",
                )
            )
        elif which == self.TENSOR_NORM_ONLY:
            mean = [float(x.strip()) for x in str(p["norm_mean"]).split(",")]
            std = [float(x.strip()) for x in str(p["norm_std"]).split(",")]
            base.append(T.Normalize(mean=mean, std=std))
        else:
            raise ValueError(f"Unknown tensor sentinel: {which}")

        return T.Compose(base)


class CodeGenerator:
    """
    Generate the torchvision v2 Compose python code from parameters.
    """

    def to_code(self, p: Dict[str, Any]) -> str:
        """
        Create a code snippet reflecting the current pipeline in interface.

        :param p: Parameters dict.
        :type p: Dict[str, Any]
        :return: Python code snippet.
        :rtype: str
        """
        lines: List[str] = [
            "from torchvision.transforms import v2 as T",
            "import torch",
            "",
            "transform = T.Compose([",
        ]

        def add(s: str) -> None:
            lines.append(f"    {s},")

        #  Geometric
        if p["use_pad"]:
            add(
                f"T.Pad(padding={int(p['pad_px'])}, fill={int(p['pad_fill'])}, padding_mode='{p['pad_mode']}')"
            )
        if p["use_resize"]:
            add(f"T.Resize(({int(p['resize_size'])}, {int(p['resize_size'])}))")
        if p["use_center_crop"]:
            add(f"T.CenterCrop(({int(p['crop_size'])}, {int(p['crop_size'])}))")
        if p["use_five_crop"]:
            add(
                f"T.FiveCrop(({int(p['five_crop_size'])}, {int(p['five_crop_size'])}))  # returns 5 images"
            )
        if p["use_random_perspective"]:
            add(
                f"T.RandomPerspective(distortion_scale={float(p['persp_dist']):.2f}, p={float(p['persp_p']):.2f})"
            )
        if p["use_random_rotation"]:
            add(f"T.RandomRotation(degrees={int(p['rot_deg'])})")
        if p["use_random_affine"]:
            add(
                "T.RandomAffine("
                f"degrees={int(p['aff_deg'])}, "
                f"translate=({float(p['aff_translate']):.2f}, {float(p['aff_translate']):.2f}), "
                f"scale=({float(p['aff_scale_min']):.2f}, {float(p['aff_scale_max']):.2f}), "
                f"shear={int(p['aff_shear'])}"
                ")"
            )
        if p["use_elastic"]:
            add(
                f"T.ElasticTransform(alpha={float(p['elastic_alpha']):.2f}, sigma={float(p['elastic_sigma']):.2f})"
            )
        if p["use_random_crop"]:
            add(
                f"T.RandomCrop(({int(p['rand_crop_size'])}, {int(p['rand_crop_size'])}))"
            )
        if p["use_rrc"]:
            add(
                "T.RandomResizedCrop("
                f"({int(p['rrc_size'])}, {int(p['rrc_size'])}), "
                f"scale=({float(p['rrc_scale_min']):.3f}, {float(p['rrc_scale_max']):.3f})"
                ")"
            )

        # Photometric
        if p["use_grayscale"]:
            add(f"T.Grayscale(num_output_channels={int(p['gray_channels'])})")
        if p["use_cj"]:
            add(
                "T.ColorJitter("
                f"brightness={float(p['cj_b']):.2f}, contrast={float(p['cj_c']):.2f}, "
                f"saturation={float(p['cj_s']):.2f}, hue={float(p['cj_h']):.2f}"
                ")"
            )
        if p["use_blur"]:
            k = int(p["blur_k"])
            if k % 2 == 0:
                k += 1
            add(
                f"T.GaussianBlur(kernel_size={k}, sigma=({float(p['blur_sigma_min']):.2f}, {float(p['blur_sigma_max']):.2f}))"
            )
        if p["use_inv"]:
            add(f"T.RandomInvert(p={float(p['inv_p']):.2f})")
        if p["use_post"]:
            add(
                f"T.RandomPosterize(bits={int(p['post_bits'])}, p={float(p['post_p']):.2f})"
            )
        if p["use_sol"]:
            add(
                f"T.RandomSolarize(threshold={int(p['sol_thresh'])}, p={float(p['sol_p']):.2f})"
            )
        if p["use_sharp"]:
            add(
                f"T.RandomAdjustSharpness(sharpness_factor={float(p['sharp_factor']):.2f}, p={float(p['sharp_p']):.2f})"
            )
        if p["use_autoc"]:
            add("T.RandomAutocontrast()")
        if p["use_eq"]:
            add("T.RandomEqualize()")
        if p["use_jpeg"]:
            add(f"T.JPEG(quality=({int(p['jpeg_qmin'])}, {int(p['jpeg_qmax'])}))")

        # Policies
        if p["use_autoaugment"]:
            add(f"T.AutoAugment(policy=T.AutoAugmentPolicy.{p['aa_policy']})")
        if p["use_randaugment"]:
            add(
                f"T.RandAugment(num_ops={int(p['ra_num_ops'])}, magnitude={int(p['ra_mag'])})"
            )
        if p["use_trivial"]:
            add(f"T.TrivialAugmentWide(num_magnitude_bins={int(p['tw_bins'])})")
        if p["use_augmix"]:
            add(
                "T.AugMix("
                f"severity={int(p['am_severity'])}, mixture_width={int(p['am_width'])}, "
                f"chain_depth={int(p['am_depth'])}, alpha={float(p['am_alpha']):.2f}"
                ")"
            )

        # Randomly-applied
        if p["use_hflip"]:
            add(f"T.RandomHorizontalFlip(p={float(p['hflip_p']):.2f})")
        if p["use_vflip"]:
            add(f"T.RandomVerticalFlip(p={float(p['vflip_p']):.2f})")
        if p["use_random_apply"]:
            add(
                f"T.RandomApply(transforms=[T.RandomCrop(({int(p['ra_crop'])}, {int(p['ra_crop'])}))], p={float(p['ra_p']):.2f})"
            )

        # Tensor-only
        need_tensor = p["use_erase"] or p["use_norm"]
        if need_tensor:
            add("T.ToImage()")
            add("T.ToDtype(torch.float32, scale=True)")

            if p["use_erase"]:
                add(
                    "T.RandomErasing("
                    f"p={float(p['erase_p']):.2f}, "
                    f"scale=({float(p['erase_scale_min']):.3f}, {float(p['erase_scale_max']):.3f}), "
                    f"ratio=({float(p['erase_ratio_min']):.2f}, {float(p['erase_ratio_max']):.2f}), "
                    'value="random"'
                    ")"
                )

            if p["use_norm"]:
                add(
                    f"T.Normalize(mean=[{p['norm_mean']}], std=[{p['norm_std']}])  # CSV -> list"
                )

        lines.append("])")
        return "\n".join(lines)


class TransformationEngine:
    """
    Apply transforms:
    - one example per enabled transform
    - final MIX pipeline with N variants (define by user)

    :param factory: TransformFactory instance.
    :type factory: TransformFactory
    """

    def __init__(self, factory: TransformFactory) -> None:
        self.factory = factory

    def apply(
        self,
        gallery_in: Any,
        n_variants: int,
        seed: int,
        reseed_each_variant: bool,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Apply transformations and return HTML.

        :param gallery_in: Gradio gallery value.
        :type gallery_in: Any
        :param n_variants: Number of variants for the MIX pipeline.
        :type n_variants: int
        :param seed: Base random seed.
        :type seed: int
        :param reseed_each_variant: Whether to re-seed each variant for reproducibility.
        :type reseed_each_variant: bool
        :param params: Transform parameters.
        :type params: Dict[str, Any]
        :return: Rendered HTML results.
        :rtype: str
        """
        images = as_pil_list(gallery_in)
        if not images:
            return ""

        base_seed = int(seed)
        singles = self.factory.build_single_transforms(params)
        grouped: Dict[int, Dict[str, Any]] = {}

        for idx, img in enumerate(images):
            grouped[idx] = {"original": img, "singles": [], "mix": []}

            # one example per transform
            for item in singles:
                tname, tform = item.name, item.op

                s = base_seed + idx * 10_000 + (abs(hash(tname)) % 10_000)
                random.seed(s)
                torch.manual_seed(s)

                if tname == "FiveCrop":
                    y = tform(img)  # tuple of 5
                    for i, crop in enumerate(y):
                        grouped[idx]["singles"].append(
                            (f"FiveCrop #{i + 1}", ensure_pil(crop))
                        )
                    continue

                if tform == TransformFactory.TENSOR_ERASE_ONLY:
                    tt = self.factory.tensor_only_example(
                        params, TransformFactory.TENSOR_ERASE_ONLY
                    )
                    grouped[idx]["singles"].append((tname, ensure_pil(tt(img))))
                    continue

                if tform == TransformFactory.TENSOR_NORM_ONLY:
                    tt = self.factory.tensor_only_example(
                        params, TransformFactory.TENSOR_NORM_ONLY
                    )
                    grouped[idx]["singles"].append((tname, ensure_pil(tt(img))))
                    continue

                grouped[idx]["singles"].append((tname, ensure_pil(tform(img))))

            # + MIX
            pipe = self.factory.build_compose(params)

            for v in range(int(n_variants)):
                if reseed_each_variant:
                    s = base_seed + idx * 1000 + v
                    random.seed(s)
                    torch.manual_seed(s)

                y = pipe(img)

                # FiveCrop inside Compose returns tuple - for mix we show first crop
                if isinstance(y, (tuple, list)) and len(y) > 0:
                    grouped[idx]["mix"].append(
                        (f"aug #{v + 1} (FiveCrop→#1)", ensure_pil(y[0]))
                    )
                else:
                    grouped[idx]["mix"].append((f"aug #{v + 1}", ensure_pil(y)))

        out = []
        for idx, block in grouped.items():
            out.append(
                {
                    "idx": idx,
                    "original": block["original"],
                    "singles": block.get("singles", []),  # list[(name, PIL)]
                    "mix": block.get("mix", []),  # list[(cap, PIL)]
                }
            )
        return out


# App logic


class TTPApp:
    """
    Main Gradio application class.

    :param i18n: I18N manager.
    :type i18n: I18N
    :param engine: Transformation engine.
    :type engine: TransformationEngine
    :param codegen: Code generator.
    :type codegen: CodeGenerator
    """

    def __init__(
        self, i18n: I18N, engine: TransformationEngine, codegen: CodeGenerator
    ) -> None:
        self.i18n = i18n
        self.engine = engine
        self.codegen = codegen

        # cache (for future?)
        # self._cache_key = None
        # self._cache_singles = None
        # self._cache_pipe = None

        # populated when building UI
        self._toggles: List[gr.Checkbox] = []
        self._params_inputs: List[gr.components.Component] = []

    def _active_sections_html(self, lang: str, p: Dict[str, Any]) -> str:
        """
        Build the "active sections" status block.

        :param p: Parameters dict.
        :type p: Dict[str, Any]
        :return: HTML snippet.
        :rtype: str
        """
        active_geo = any(
            p[k]
            for k in [
                "use_pad",
                "use_resize",
                "use_center_crop",
                "use_five_crop",
                "use_random_perspective",
                "use_random_rotation",
                "use_random_affine",
                "use_elastic",
                "use_random_crop",
                "use_rrc",
            ]
        )
        active_photo = any(
            p[k]
            for k in [
                "use_grayscale",
                "use_cj",
                "use_blur",
                "use_inv",
                "use_post",
                "use_sol",
                "use_sharp",
                "use_autoc",
                "use_eq",
                "use_jpeg",
            ]
        )
        active_aug = any(
            p[k]
            for k in ["use_autoaugment", "use_randaugment", "use_trivial", "use_augmix"]
        )
        active_randomly = any(
            p[k] for k in ["use_hflip", "use_vflip", "use_random_apply"]
        )
        active_tensor = any(p[k] for k in ["use_erase", "use_norm"])

        return f"""
        <div style="font-size:14px;line-height:1.6">
          <div>{status_dot(active_geo)}<b>{self.i18n.section(lang, 'geometric')}</b></div>
          <div>{status_dot(active_photo)}<b>{self.i18n.section(lang, 'photometric')}</b></div>
          <div>{status_dot(active_aug)}<b>{self.i18n.section(lang, 'policies')}</b></div>
          <div>{status_dot(active_randomly)}<b>{self.i18n.section(lang, 'random_applied')}</b></div>
          <div>{status_dot(active_tensor)}<b>{self.i18n.section(lang, 'tensor_bonus')}</b></div>
        </div>
        """

    def _collect_params(self, *vals: Any) -> Dict[str, Any]:
        """
        Collect UI values into a params dict (order must match self._params_inputs).

        :param vals: Values from Gradio components.
        :type vals: Any
        :return: Parameters dict.
        :rtype: Dict[str, Any]
        """
        keys = [
            # Geometric
            "use_pad",
            "pad_px",
            "pad_fill",
            "pad_mode",
            "use_resize",
            "resize_size",
            "use_center_crop",
            "crop_size",
            "use_five_crop",
            "five_crop_size",
            "use_random_perspective",
            "persp_p",
            "persp_dist",
            "use_random_rotation",
            "rot_deg",
            "use_random_affine",
            "aff_deg",
            "aff_translate",
            "aff_scale_min",
            "aff_scale_max",
            "aff_shear",
            "use_elastic",
            "elastic_alpha",
            "elastic_sigma",
            "use_random_crop",
            "rand_crop_size",
            "use_rrc",
            "rrc_size",
            "rrc_scale_min",
            "rrc_scale_max",
            # Photometric
            "use_grayscale",
            "gray_channels",
            "use_cj",
            "cj_b",
            "cj_c",
            "cj_s",
            "cj_h",
            "use_blur",
            "blur_k",
            "blur_sigma_min",
            "blur_sigma_max",
            "use_inv",
            "inv_p",
            "use_post",
            "post_p",
            "post_bits",
            "use_sol",
            "sol_p",
            "sol_thresh",
            "use_sharp",
            "sharp_p",
            "sharp_factor",
            "use_autoc",
            "use_eq",
            "use_jpeg",
            "jpeg_qmin",
            "jpeg_qmax",
            # Policies
            "use_autoaugment",
            "aa_policy",
            "use_randaugment",
            "ra_num_ops",
            "ra_mag",
            "use_trivial",
            "tw_bins",
            "use_augmix",
            "am_severity",
            "am_width",
            "am_depth",
            "am_alpha",
            # Randomly-applied
            "use_hflip",
            "hflip_p",
            "use_vflip",
            "vflip_p",
            "use_random_apply",
            "ra_p",
            "ra_crop",
            # Tensor-only, bonus
            "use_erase",
            "erase_p",
            "erase_scale_min",
            "erase_scale_max",
            "erase_ratio_min",
            "erase_ratio_max",
            "use_norm",
            "norm_mean",
            "norm_std",
        ]

        p = dict(zip(keys, vals))

        # Safety: ensure bool toggles are bool
        for k in list(p.keys()):
            if k.startswith("use_"):
                p[k] = bool(p[k])

        return p

    def _disable_all(self) -> List[gr.update]:
        """
        Disable all transform toggles.

        :return: List of gr.update objects setting each toggle to False.
        :rtype: List[gr.update]
        """
        return [gr.update(value=False) for _ in self._toggles]

    def _set_language(self, lang: str):
        # Markdown ONLY (pas de <h1>, pas de <div>)
        title = f"# {self.i18n.get(lang, 'app_title')}"
        desc = f"{self.i18n.subtitles(lang)[0]}"
        return gr.update(value=title), gr.update(value=desc)

    def build(self) -> gr.Blocks:
        """
        Build the Gradio interface and wire callbacks.

        :return: Gradio Blocks app.
        :rtype: gradio.Blocks
        """
        i18n = self.i18n

        # Documentation links (simple + stable)
        DOCS = {
            "geometric": "https://pytorch.org/vision/stable/transforms.html#geometry",
            "photometric": "https://pytorch.org/vision/stable/transforms.html#color",
            "policies": "https://docs.pytorch.org/vision/stable/transforms.html#id8",
            "random_applied": "https://pytorch.org/vision/stable/transforms.html",
            "tensor_bonus": "https://docs.pytorch.org/vision/stable/transforms.html#id6",
        }

        with gr.Blocks(title=i18n.get("EN", "app_title")) as demo:
            # Header (centered title + subtitle + language)

            title_md = gr.Markdown(value=f"# {i18n.get('EN', 'app_title')}")
            desc_md = gr.Markdown(value=f"### {i18n.subtitles('EN')[0]}")

            with gr.Sidebar(open=True, width=550, elem_id="controls_sidebar"):
                globals_title = gr.Markdown(f"### {i18n.get('EN', 'globals_title')}")
                lang = gr.Radio(
                    ["EN", "FR"], value="EN", label=i18n.get("EN", "language_label")
                )
                n_variants = gr.Slider(
                    1, 8, value=3, step=1, label=i18n.get("EN", "variants_label")
                )
                seed = gr.Number(
                    value=42, precision=0, label=i18n.get("EN", "seed_label")
                )
                reseed_each_variant = gr.Checkbox(
                    value=True, label=i18n.get("EN", "reseed_label")
                )

                disable_all_btn = gr.Button(i18n.get("EN", "disable_all"))
                status_html = gr.HTML(label=i18n.get("EN", "status_label"))
                code_preview = gr.Code(
                    label=i18n.get("EN", "code_label"), language="python"
                )

                # Geometric
                acc_geo = gr.Accordion(
                    label=i18n.section("EN", "geometric"), open=False
                )
                with acc_geo:
                    docs_geo_md = gr.Markdown(
                        f"{i18n.get('EN', 'docs_prefix')} [{DOCS['geometric']}]({DOCS['geometric']})"
                    )

                    use_pad = gr.Checkbox(value=False, label="Pad")
                    pad_px = gr.Slider(0, 200, value=20, step=1, label="padding (px)")
                    pad_fill = gr.Slider(0, 255, value=0, step=1, label="fill (0-255)")
                    pad_mode = gr.Dropdown(
                        ["constant", "edge", "reflect", "symmetric"],
                        value="constant",
                        label="padding_mode",
                    )

                    use_resize = gr.Checkbox(value=False, label="Resize (square)")
                    resize_size = gr.Slider(
                        64, 1024, value=256, step=1, label="Resize size"
                    )

                    use_center_crop = gr.Checkbox(value=False, label="CenterCrop")
                    crop_size = gr.Slider(
                        32, 1024, value=224, step=1, label="Crop size"
                    )

                    use_five_crop = gr.Checkbox(
                        value=False, label="FiveCrop (shows 5 images)"
                    )
                    five_crop_size = gr.Slider(
                        32, 1024, value=224, step=1, label="FiveCrop size"
                    )

                    use_random_perspective = gr.Checkbox(
                        value=False, label="RandomPerspective"
                    )
                    persp_p = gr.Slider(0, 1, value=0.5, step=0.05, label="p")
                    persp_dist = gr.Slider(
                        0, 1, value=0.5, step=0.05, label="distortion_scale"
                    )

                    use_random_rotation = gr.Checkbox(
                        value=False, label="RandomRotation"
                    )
                    rot_deg = gr.Slider(0, 180, value=15, step=1, label="degrees")

                    use_random_affine = gr.Checkbox(value=False, label="RandomAffine")
                    aff_deg = gr.Slider(0, 180, value=15, step=1, label="degrees")
                    aff_translate = gr.Slider(
                        0, 0.5, value=0.1, step=0.01, label="translate (fraction)"
                    )
                    aff_scale_min = gr.Slider(
                        0.1, 2.0, value=0.9, step=0.05, label="scale min"
                    )
                    aff_scale_max = gr.Slider(
                        0.1, 2.0, value=1.1, step=0.05, label="scale max"
                    )
                    aff_shear = gr.Slider(0, 45, value=10, step=1, label="shear (deg)")

                    use_elastic = gr.Checkbox(value=False, label="ElasticTransform")
                    elastic_alpha = gr.Slider(
                        0.0, 200.0, value=50.0, step=1.0, label="alpha"
                    )
                    elastic_sigma = gr.Slider(
                        0.0, 50.0, value=5.0, step=0.5, label="sigma"
                    )

                    use_random_crop = gr.Checkbox(value=False, label="RandomCrop")
                    rand_crop_size = gr.Slider(
                        32, 1024, value=224, step=1, label="RandomCrop size"
                    )

                    use_rrc = gr.Checkbox(value=False, label="RandomResizedCrop")
                    rrc_size = gr.Slider(32, 1024, value=224, step=1, label="RRC size")
                    rrc_scale_min = gr.Slider(
                        0.05, 1.0, value=0.5, step=0.01, label="RRC scale min"
                    )
                    rrc_scale_max = gr.Slider(
                        0.05, 1.0, value=1.0, step=0.01, label="RRC scale max"
                    )

                # Photometric
                acc_photo = gr.Accordion(
                    label=i18n.section("EN", "photometric"), open=True
                )
                with acc_photo:
                    docs_photo_md = gr.Markdown(
                        f"{i18n.get('EN', 'docs_prefix')} [{DOCS['photometric']}]({DOCS['photometric']})"
                    )
                    use_grayscale = gr.Checkbox(value=False, label="Grayscale")
                    gray_channels = gr.Radio(
                        [1, 3], value=3, label="num_output_channels"
                    )

                    use_cj = gr.Checkbox(value=True, label="ColorJitter")
                    cj_b = gr.Slider(0, 2, value=0.2, step=0.05, label="brightness")
                    cj_c = gr.Slider(0, 2, value=0.2, step=0.05, label="contrast")
                    cj_s = gr.Slider(0, 2, value=0.2, step=0.05, label="saturation")
                    cj_h = gr.Slider(0, 0.5, value=0.05, step=0.01, label="hue")

                    use_blur = gr.Checkbox(value=False, label="GaussianBlur")
                    blur_k = gr.Slider(
                        1, 61, value=11, step=2, label="kernel_size (odd)"
                    )
                    blur_sigma_min = gr.Slider(
                        0.1, 10.0, value=0.1, step=0.1, label="sigma min"
                    )
                    blur_sigma_max = gr.Slider(
                        0.1, 10.0, value=2.0, step=0.1, label="sigma max"
                    )

                    use_inv = gr.Checkbox(value=True, label="RandomInvert")
                    inv_p = gr.Slider(0, 1, value=0.50, step=0.05, label="p")

                    use_post = gr.Checkbox(value=False, label="RandomPosterize")
                    post_p = gr.Slider(0, 1, value=0.2, step=0.05, label="p")
                    post_bits = gr.Slider(1, 8, value=4, step=1, label="bits")

                    use_sol = gr.Checkbox(value=True, label="RandomSolarize")
                    sol_p = gr.Slider(0, 1, value=0.40, step=0.05, label="p")
                    sol_thresh = gr.Slider(
                        0, 255, value=128, step=1, label="threshold (0-255)"
                    )

                    use_sharp = gr.Checkbox(value=False, label="RandomAdjustSharpness")
                    sharp_p = gr.Slider(0, 1, value=0.5, step=0.05, label="p")
                    sharp_factor = gr.Slider(
                        0.0, 5.0, value=2.0, step=0.1, label="sharpness_factor"
                    )

                    use_autoc = gr.Checkbox(value=True, label="RandomAutocontrast")
                    use_eq = gr.Checkbox(value=True, label="RandomEqualize")

                    use_jpeg = gr.Checkbox(value=False, label="JPEG (compression)")
                    jpeg_qmin = gr.Slider(1, 100, value=5, step=1, label="quality min")
                    jpeg_qmax = gr.Slider(1, 100, value=50, step=1, label="quality max")

                # Policies
                acc_policies = gr.Accordion(
                    label=i18n.section("EN", "policies"), open=False
                )
                with acc_policies:
                    docs_acc_md = gr.Markdown(
                        f"{i18n.get('EN', 'docs_prefix')} [{DOCS['policies']}]({DOCS['policies']})"
                    )

                    use_autoaugment = gr.Checkbox(value=False, label="AutoAugment")
                    aa_policy = gr.Dropdown(
                        ["CIFAR10", "IMAGENET", "SVHN"],
                        value="IMAGENET",
                        label="policy",
                    )

                    use_randaugment = gr.Checkbox(value=False, label="RandAugment")
                    ra_num_ops = gr.Slider(1, 10, value=2, step=1, label="num_ops")
                    ra_mag = gr.Slider(0, 30, value=9, step=1, label="magnitude")

                    use_trivial = gr.Checkbox(value=False, label="TrivialAugmentWide")
                    tw_bins = gr.Slider(
                        1, 50, value=31, step=1, label="num_magnitude_bins"
                    )

                    use_augmix = gr.Checkbox(value=False, label="AugMix")
                    am_severity = gr.Slider(1, 10, value=3, step=1, label="severity")
                    am_width = gr.Slider(1, 10, value=3, step=1, label="mixture_width")
                    am_depth = gr.Slider(
                        -1, 10, value=-1, step=1, label="chain_depth (-1 = random)"
                    )
                    am_alpha = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="alpha")

                # Randomly-applied
                acc_random = gr.Accordion(
                    label=i18n.section("EN", "random_applied"), open=True
                )
                with acc_random:
                    docs_random_md = gr.Markdown(
                        f"{i18n.get('EN', 'docs_prefix')} [{DOCS['random_applied']}]({DOCS['random_applied']})"
                    )

                    use_hflip = gr.Checkbox(value=True, label="RandomHorizontalFlip")
                    hflip_p = gr.Slider(0, 1, value=0.5, step=0.05, label="p")

                    use_vflip = gr.Checkbox(value=False, label="RandomVerticalFlip")
                    vflip_p = gr.Slider(0, 1, value=0.2, step=0.05, label="p")

                    use_random_apply = gr.Checkbox(
                        value=False, label="RandomApply([RandomCrop])"
                    )
                    ra_p = gr.Slider(0, 1, value=0.5, step=0.05, label="p")
                    ra_crop = gr.Slider(
                        32, 1024, value=64, step=1, label="inner RandomCrop size"
                    )

                # Tensor-only (bonus)
                acc_tensor = gr.Accordion(
                    label=i18n.section("EN", "tensor_bonus"), open=False
                )
                with acc_tensor:
                    docs_tensor_md = gr.Markdown(
                        f"{i18n.get('EN', 'docs_prefix')} [{DOCS['tensor_bonus']}]({DOCS['tensor_bonus']})"
                    )

                    use_erase = gr.Checkbox(value=False, label="RandomErasing")
                    erase_p = gr.Slider(0, 1, value=0.25, step=0.05, label="p")
                    erase_scale_min = gr.Slider(
                        0.0001, 0.5, value=0.02, step=0.01, label="scale min"
                    )
                    erase_scale_max = gr.Slider(
                        0.0001, 1.0, value=0.2, step=0.01, label="scale max"
                    )
                    erase_ratio_min = gr.Slider(
                        0.1, 10.0, value=0.3, step=0.1, label="ratio min"
                    )
                    erase_ratio_max = gr.Slider(
                        0.1, 10.0, value=3.3, step=0.1, label="ratio max"
                    )

                    use_norm = gr.Checkbox(value=False, label="Normalize (mean/std)")
                    norm_mean = gr.Textbox(
                        value="0.485,0.456,0.406", label="mean (CSV)"
                    )
                    norm_std = gr.Textbox(value="0.229,0.224,0.225", label="std (CSV)")

            # Main content
            with gr.Column(scale=9):
                upload_title_md = gr.Markdown(f"## {i18n.get('EN', 'upload_section')}")
                gallery_in = gr.Gallery(
                    label=i18n.get("EN", "upload_label"),
                    type="pil",
                    columns=4,
                    height=240,
                )

                apply_btn = gr.Button(i18n.get("EN", "apply"), variant="primary")
                results_state = gr.State([])
                results_title_md = gr.Markdown(
                    f"## {i18n.get('EN', 'results_section')}"
                )

                @gr.render(inputs=results_state)
                def render_results(data: Any) -> None:
                    """
                    Render the results accordions + galleries.
                    :param data: Data from results_state.
                    :type data: Any
                    :return: None
                    :rtype: None
                    """
                    if not data:
                        gr.Markdown("")
                        return

                    for item in data:
                        with gr.Accordion(label=f"Image #{item['idx']}", open=True):
                            # 1 container == 1 gallery (original + singles + mix)
                            tiles = []
                            tiles.append((item["original"], "Original"))

                            # singles: list[(name, PIL)]
                            tiles += [
                                (im, name) for (name, im) in item.get("singles", [])
                            ]

                            # mix: list[(cap, PIL)]
                            tiles += [(im, cap) for (cap, im) in item.get("mix", [])]

                            gr.Gallery(
                                value=tiles,
                                label=None,
                                columns=4,
                                height=260,
                                preview=True,
                            )

                # use this to disable all
                self._toggles = [
                    use_pad,
                    use_resize,
                    use_center_crop,
                    use_five_crop,
                    use_random_perspective,
                    use_random_rotation,
                    use_random_affine,
                    use_elastic,
                    use_random_crop,
                    use_rrc,
                    use_grayscale,
                    use_cj,
                    use_blur,
                    use_inv,
                    use_post,
                    use_sol,
                    use_sharp,
                    use_autoc,
                    use_eq,
                    use_jpeg,
                    use_autoaugment,
                    use_randaugment,
                    use_trivial,
                    use_augmix,
                    use_hflip,
                    use_vflip,
                    use_random_apply,
                    use_erase,
                    use_norm,
                ]

                self._params_inputs = [
                    use_pad,
                    pad_px,
                    pad_fill,
                    pad_mode,
                    use_resize,
                    resize_size,
                    use_center_crop,
                    crop_size,
                    use_five_crop,
                    five_crop_size,
                    use_random_perspective,
                    persp_p,
                    persp_dist,
                    use_random_rotation,
                    rot_deg,
                    use_random_affine,
                    aff_deg,
                    aff_translate,
                    aff_scale_min,
                    aff_scale_max,
                    aff_shear,
                    use_elastic,
                    elastic_alpha,
                    elastic_sigma,
                    use_random_crop,
                    rand_crop_size,
                    use_rrc,
                    rrc_size,
                    rrc_scale_min,
                    rrc_scale_max,
                    use_grayscale,
                    gray_channels,
                    use_cj,
                    cj_b,
                    cj_c,
                    cj_s,
                    cj_h,
                    use_blur,
                    blur_k,
                    blur_sigma_min,
                    blur_sigma_max,
                    use_inv,
                    inv_p,
                    use_post,
                    post_p,
                    post_bits,
                    use_sol,
                    sol_p,
                    sol_thresh,
                    use_sharp,
                    sharp_p,
                    sharp_factor,
                    use_autoc,
                    use_eq,
                    use_jpeg,
                    jpeg_qmin,
                    jpeg_qmax,
                    use_autoaugment,
                    aa_policy,
                    use_randaugment,
                    ra_num_ops,
                    ra_mag,
                    use_trivial,
                    tw_bins,
                    use_augmix,
                    am_severity,
                    am_width,
                    am_depth,
                    am_alpha,
                    use_hflip,
                    hflip_p,
                    use_vflip,
                    vflip_p,
                    use_random_apply,
                    ra_p,
                    ra_crop,
                    use_erase,
                    erase_p,
                    erase_scale_min,
                    erase_scale_max,
                    erase_ratio_min,
                    erase_ratio_max,
                    use_norm,
                    norm_mean,
                    norm_std,
                ]

                # this for live updates of status + code
                def _update_all(lang_val: str, *vals: Any) -> Tuple[str, str]:
                    """
                    Update status HTML + code preview.

                    :param lang_val: language code.
                    :type lang_val: str
                    :param vals: values from Gradio components.
                    :type vals: Any
                    :return: Tuple of (status HTML, code preview).
                    :rtype: Tuple[str, str]
                    """
                    p = self._collect_params(*vals)
                    status = self._active_sections_html(lang_val, p)
                    code = self.codegen.to_code(p)
                    return status, code

                for comp in self._params_inputs:
                    comp.change(
                        fn=_update_all,
                        inputs=[lang] + self._params_inputs,
                        outputs=[status_html, code_preview],
                    )

                demo.load(
                    fn=_update_all,
                    inputs=[lang] + self._params_inputs,
                    outputs=[status_html, code_preview],
                )

                disable_all_btn.click(
                    fn=self._disable_all,
                    inputs=[],
                    outputs=self._toggles,
                )

                # --- apply button ---
                def _apply(
                    gallery: Any, nvar: int, sd: int, reseed: bool, *vals: Any
                ) -> str:
                    """
                    Apply transformations.

                    :param gallery: the input gallery.
                    :type gallery: Any
                    :param nvar: number of variants.
                    :type nvar: int
                    :param sd: seed.
                    :type sd: int
                    :param reseed: whether to reseed each variant.
                    :type reseed: bool
                    :param vals: values from Gradio components.
                    :type vals: Any
                    :return: Rendered HTML results.
                    :rtype: str
                    """
                    p = self._collect_params(*vals)

                    return self.engine.apply(
                        gallery, int(nvar), int(sd), bool(reseed), p
                    )

                apply_btn.click(
                    fn=_apply,
                    inputs=[gallery_in, n_variants, seed, reseed_each_variant]
                    + self._params_inputs,
                    outputs=[results_state],
                )

                def _on_lang_change(lang_val: str, *vals: Any):
                    """
                    Handle language change: update UI + status + code.

                    :param lang_val: language code.
                    :type lang_val: str
                    :param vals: values from Gradio components.
                    :type vals: Any
                    :return: Updated components.
                    :rtype: Tuple[gr.update, ...]
                    """
                    # updates UI
                    t_upd, desc_upd = self._set_language(lang_val)

                    # status recalculation
                    p = self._collect_params(*vals)
                    status = self._active_sections_html(lang_val, p)
                    code = self.codegen.to_code(p)

                    return (
                        # header
                        t_upd,
                        desc_upd,
                        # globals
                        gr.update(value=f"### {i18n.get(lang_val, 'globals_title')}"),
                        gr.update(label=i18n.get(lang_val, "language_label")),
                        gr.update(value=i18n.get(lang_val, "disable_all")),
                        gr.update(label=i18n.get(lang_val, "variants_label")),
                        gr.update(label=i18n.get(lang_val, "seed_label")),
                        gr.update(label=i18n.get(lang_val, "reseed_label")),
                        gr.update(value=i18n.get(lang_val, "apply")),
                        gr.update(label=i18n.get(lang_val, "upload_label")),
                        # section titles
                        gr.update(value=f"## {i18n.get(lang_val, 'upload_section')}"),
                        gr.update(value=f"## {i18n.get(lang_val, 'results_section')}"),
                        # accordions labels
                        gr.update(label=i18n.section(lang_val, "geometric")),
                        gr.update(label=i18n.section(lang_val, "photometric")),
                        gr.update(label=i18n.section(lang_val, "policies")),
                        gr.update(label=i18n.section(lang_val, "random_applied")),
                        gr.update(label=i18n.section(lang_val, "tensor_bonus")),
                        # docs prefix markdowns
                        gr.update(
                            value=f"{i18n.get(lang_val, 'docs_prefix')} [{DOCS['geometric']}]({DOCS['geometric']})"
                        ),
                        gr.update(
                            value=f"{i18n.get(lang_val, 'docs_prefix')} [{DOCS['photometric']}]({DOCS['photometric']})"
                        ),
                        gr.update(
                            value=f"{i18n.get(lang_val, 'docs_prefix')} [{DOCS['policies']}]({DOCS['policies']})"
                        ),
                        gr.update(
                            value=f"{i18n.get(lang_val, 'docs_prefix')} [{DOCS['random_applied']}]({DOCS['random_applied']})"
                        ),
                        gr.update(
                            value=f"{i18n.get(lang_val, 'docs_prefix')} [{DOCS['tensor_bonus']}]({DOCS['tensor_bonus']})"
                        ),
                        # ✅ status + code recalculés
                        gr.update(value=status),
                        gr.update(value=code),
                    )

                # When language changes: update title/subtitle + some labels
                lang.change(
                    fn=_on_lang_change,
                    inputs=[lang] + self._params_inputs,
                    outputs=[
                        # header
                        title_md,
                        desc_md,
                        # globals
                        globals_title,
                        lang,
                        disable_all_btn,
                        n_variants,
                        seed,
                        reseed_each_variant,
                        apply_btn,
                        gallery_in,
                        # section titles
                        upload_title_md,
                        results_title_md,
                        # accordion labels
                        acc_geo,
                        acc_photo,
                        acc_policies,
                        acc_random,
                        acc_tensor,
                        # docs markdowns
                        docs_geo_md,
                        docs_photo_md,
                        docs_acc_md,
                        docs_random_md,
                        docs_tensor_md,
                        # status + code gen
                        status_html,
                        code_preview,
                    ],
                )

            return demo


def main() -> None:
    """
    Gradio entrypoint.

    :return: None
    :rtype: None
    """
    texts_json = load_texts_json(DEFAULT_I18N_PATH)
    i18n = I18N(texts_json, default_lang="EN")
    factory = TransformFactory()
    codegen = CodeGenerator()
    engine = TransformationEngine(factory)
    app = TTPApp(i18n=i18n, engine=engine, codegen=codegen)

    demo = app.build()
    demo.launch(css=load_css(DEFAULT_CSS_PATH))


if __name__ == "__main__":
    main()
