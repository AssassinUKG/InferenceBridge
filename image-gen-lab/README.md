# InferenceBridge Image Generation Lab

This folder is an isolated quality and feasibility test for local image
generation. It does not change or call the main InferenceBridge application.

The first candidate is Qwen-Image-2512 with:

- a Q6_K diffusion transformer;
- a Q8_0 Qwen2.5-VL-7B text encoder;
- the original Qwen Image VAE;
- a pinned CUDA build of `stable-diffusion.cpp`.

Every downloaded file has a pinned repository revision, byte size and SHA-256
checksum in `config/manifest.json`. Downloads resume after interruption.

## Commands

Run these from the InferenceBridge repository root:

```powershell
python image-gen-lab/lab.py validate
python image-gen-lab/lab.py install-runtime
python image-gen-lab/lab.py download-models --bundle qwen-2512-q6-quality
python image-gen-lab/lab.py doctor --bundle qwen-2512-q6-quality
python image-gen-lab/lab.py generate --profile smoke --prompt-id surfboard-cat
python image-gen-lab/lab.py generate --profile quality-1024 --prompt-id surfboard-cat
python image-gen-lab/lab.py benchmark --profile quality-1024 --suite quality-core
python image-gen-lab/lab.py reanalyze
python image-gen-lab/lab.py report
python -m unittest discover -s image-gen-lab/tests -v
```

Use `--dry-run` with `generate` or `benchmark` to inspect the exact native
runner arguments without loading a model.

## Storage

The core Q6 bundle is approximately 25.2 GB, plus the 0.4 GB CUDA runtime and
generated results. The optional Q8 transformer is not part of the initial
download.

Downloaded components live under `artifacts/` and generated images, logs and
GPU telemetry live under `results/`. Both directories are deliberately ignored
by Git.

## Quality policy

- Q4 is not an automatic fallback for Quality mode.
- The original VAE is always used.
- Final-quality profiles do not use TAESD.
- Generation failures are reported; profiles are never silently downgraded.
- The model, prompt, seed, settings and component checksums are stored beside
  every output image.

The `smoke` profile checks compatibility at low cost. `quality-1024` is the
first visual-quality gate. `quality-native-square` uses Qwen's 1328 x 1328
native square size and is intentionally more demanding.

See `reports/ASSESSMENT.md` for the measured RTX 3090 results and integration
recommendation.
