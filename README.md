# Torchvision Transforms Playground (Gradio)


[![ruff](https://img.shields.io/badge/lint-ruff-0A0A0A?logo=ruff&logoColor=white)](https://img.shields.io/badge/lint-ruff-0A0A0A?logo=ruff&logoColor=white)
[![Gradio](https://img.shields.io/badge/Gradio-app-orange)](https://www.gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/lterriel/Torchvision-Transforms-Playground)
![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)

Interactive sandbox to explore torchvision transforms on your images:
- Upload one or multiple images
- Enable/disable transforms and tweak parameters
- Preview one example per enabled transform + a final MIX pipeline (multiple random variants)
- Export a ready-to-use `torchvision.transforms.v2.Compose` code snippet
- Switch UI language (EN/FR)

## Demo on 🤗 Hugging Face Spaces

[Try it here](https://huggingface.co/spaces/lterriel/Torchvision-Transforms-Playground)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open your browser at `http://localhost:7860`.


## Citation

```
@software{terriel_torchvision_transforms_playground_2026,
  author       = {Terriel, Lucas},
  title        = {Torchvision Transforms Playground},
  year         = {2026},
  url          = {https://github.com/Lucaterre/Torchvision-Transforms-Playground},
  note         = {Gradio-based interactive sandbox for exploring torchvision image transformations},
}
```
