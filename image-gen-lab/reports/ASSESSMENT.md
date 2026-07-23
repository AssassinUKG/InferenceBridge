# Qwen-Image-2512 Q6 Feasibility Assessment

Test date: 23 July 2026

Hardware:

- NVIDIA GeForce RTX 3090, 24 GB VRAM
- AMD Ryzen 7 5800X
- 64 GB system RAM
- Windows 11

## Verdict

Qwen-Image-2512 Q6 is good enough to be InferenceBridge's first Quality image
model on this machine. The tested Q6 transformer does not show an obvious
quality problem that justifies downloading Q8 yet.

Recommended IB profiles:

| IB profile | Settings | Expected time | Observed VRAM | Recommendation |
|---|---|---:|---:|---|
| Preview | 512 x 512, 4 steps | 13-17 s | 18.8 GB | Compatibility/preview only |
| Quality | 1024 x 1024, 50 steps | 3:10-3:14 | 19.3-19.4 GB | Default |
| Max Quality | 1328 x 1328, 50 steps | 5:47 | 20.1 GB | Optional, thermally guarded |

Q4 should not be used for the Quality profile. Q8 should remain an optional
A/B candidate only if broader Q6 testing reveals a visible problem.

## Actual results

Eight independent generation jobs completed successfully:

- one initial smoke test;
- three 1024 x 1024 quality images;
- one native 1328 x 1328 quality image;
- three repeated smoke/load/unload cycles.

All outputs had the requested dimensions, valid PNG headers and unique output
checksums where different seeds were used.

The two independent smoke runs using seed `424242` produced the identical
SHA-256:

`c4790aac925ef88461d2c70a34b3497ce437e301d6be24f889dbc51c42588926`

This confirms deterministic output for the tested runtime, profile, prompt and
seed.

## Visual quality findings

### Surfboard cat

[Open 1024 quality image](../results/20260723-114023-quality-1024-surfboard-cat-seed424242/output.png)

The image has convincing wet fur, droplets, wave structure, four coherent paws,
clean surfboard geometry and believable contact between the cat and board. It
has a slightly polished stock-photo character, but no major anatomical or
composition failure.

### English text rendering

[Open cafe text image](../results/20260723-114408-quality-1024-cafe-sign-seed424242/output.png)

Both requested phrases are visibly correct:

- `THE QUIET CUP`
- `TEA, CAKE & GOOD BOOKS`

Tesseract did not recover the complete phrases from the angled whole scene, so
the conservative automatic exact-OCR score is a false negative. This result
shows why OCR can assist evaluation but cannot replace direct inspection.

### Hands and anatomy

[Open pottery image](../results/20260723-114943-quality-1024-portrait-hands-seed424242/output.png)

Both hands are present, finger counts and contact are coherent, the clay residue
is plausible, and the bowl/wheel geometry is clean. No obvious extra-finger or
merged-hand failure is visible.

### Native 1328 resolution

[Open native 1328 image](../results/20260723-115344-quality-native-square-surfboard-cat-seed424242/output.png)

The native result adds fine fur and water detail, but the improvement is modest
relative to its 1.83x runtime and higher temperature. It should be exposed as
Max Quality rather than silently replacing the 1024 profile.

## Runtime and thermal findings

The pinned CUDA runtime automatically placed the text encoder, diffusion
transformer and VAE without an out-of-memory failure. The Q6 diffusion graph fit
as a single GPU segment.

- 1024 peak VRAM: 19.34-19.42 GB
- 1328 peak VRAM: 20.07 GB
- VRAM after every run: approximately 2.4 GB baseline
- First isolated 1024 peak temperature: 81 C
- Later consecutive 1024 runs: 84 C
- Native 1328 peak temperature: 87 C

The 1328 run showed normal driver-level thermal limiting. IB should not alter
GPU power or fan settings. It should:

- show temperature while generating;
- avoid starting the next queued job while the GPU is already hot;
- warn at 85 C or above;
- wait for a configurable cooldown before unattended batch work.

## What the lab proves

- The selected files and native runtime are mutually compatible.
- Q6 quality is strong on animal detail, exact English signage and hands.
- 1024/50 fits the RTX 3090 with several gigabytes of VRAM headroom.
- Native 1328 is feasible.
- Repeated process load/generate/unload cycles clean up GPU memory.
- Fixed seed output is deterministic.
- The model bundle can be downloaded and updated component by component using
  pinned revisions and checksums.

## What remains for IB integration

This lab deliberately does not change the main InferenceBridge runtime.
Integration still needs:

1. A shared GPU lock and chat-to-image-to-chat lifecycle.
2. Guaranteed chat model restoration after success, cancellation or failure.
3. Cancellation and forced-runner-crash recovery tests.
4. Image attachment persistence in the session database.
5. A code-owned `generate_image` capability and optional real tool definition.
6. Ten complete IB chat-model swap cycles.
7. A larger blind prompt/seed gallery before deciding whether Q8 is worth its
   additional storage and offloading cost.

The evidence from this lab supports proceeding with Q6 integration without
downloading Q8 first.

