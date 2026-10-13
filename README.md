# Don’t Peak: LoRA Experiments

An archive of an unsuccessful LoRA training effort, including notebook variants, model-format conversion and SDXL/SD 1.5 experiments.

**Status:** Failed experiment, preserved as a record of the attempted approach—not a trained-model release.

## What is here

- `lora-trainer-script.ipynb`: initial training notebook.
- `sd1.5/` and `sdxl/`: retained notebook and format documentation variants.
- `converter_tocomfyui.py`, `HF_FORMAT_ONLY__AND_SDXL_MERGER.py` and `conveerters/`: conversion experiments.
- Notebook source is retained with outputs cleared.

## Getting started

Read the notebooks before running them. They depend on their original GPU environments, external base models, training images and paths. Supply your own permitted dataset and adapt the configuration.

No notebook or converter is run automatically. Model weights, checkpoints, nested dataset archives and large training records are deliberately absent.

## Current boundaries

This snapshot does not establish a reproducible successful training result. The original author classified the effort as failed; the exact failure cause has not been independently established. Some scripts contain historical path assumptions and should be treated as experimental source, not ready-to-run tools.

## Preservation and checks

This repository preserves a development snapshot, including incomplete work. See [PUBLICATION.md](PUBLICATION.md) for what was retained or excluded and [VALIDATION.md](VALIDATION.md) for the checks actually run. Original copyright and license notices take precedence over the repository’s display name.
