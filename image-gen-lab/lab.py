#!/usr/bin/env python3
"""Isolated quality benchmark for InferenceBridge image generation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable


LAB_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = LAB_ROOT / "config" / "manifest.json"
PROMPTS_PATH = LAB_ROOT / "config" / "prompts.json"
ARTIFACTS_ROOT = LAB_ROOT / "artifacts"
RESULTS_ROOT = LAB_ROOT / "results"
REPORTS_ROOT = LAB_ROOT / "reports"


class LabError(RuntimeError):
    """Expected user-facing lab failure."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_under(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing traversal outside root."""
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise LabError(f"Unsafe path outside lab root: {relative_path}") from exc
    return candidate


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, size: int, sha256: str, hash_file: bool = True) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != size:
        return False, f"size mismatch ({actual_size} != {size})"
    if hash_file:
        actual_hash = sha256_file(path)
        if actual_hash.lower() != sha256.lower():
            return False, f"SHA-256 mismatch ({actual_hash})"
    return True, "verified"


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported manifest schema_version")

    runtime = manifest.get("runtime", {})
    assets = manifest.get("assets", {})
    bundles = manifest.get("bundles", {})
    profiles = manifest.get("profiles", {})

    pinned_files = [("runtime", runtime), ("runtime.cuda_runtime", runtime.get("cuda_runtime", {}))]
    pinned_files.extend(assets.items())
    for label, item in pinned_files:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(item.get("sha256", ""))):
            errors.append(f"{label}: invalid SHA-256")
        if not isinstance(item.get("size"), int) or item["size"] <= 0:
            errors.append(f"{label}: invalid byte size")
        if not str(item.get("url", "")).startswith("https://"):
            errors.append(f"{label}: URL must use HTTPS")
        if label != "runtime":
            try:
                resolve_under(ARTIFACTS_ROOT, item.get("relative_path", ""))
            except LabError as exc:
                errors.append(f"{label}: {exc}")

    for bundle_name, bundle in bundles.items():
        for key in ("transformer", "text_encoder", "vae"):
            if bundle.get(key) not in assets:
                errors.append(f"{bundle_name}: unknown {key} asset {bundle.get(key)!r}")

    for profile_name, profile in profiles.items():
        for dimension in ("width", "height"):
            value = profile.get(dimension)
            if not isinstance(value, int) or value <= 0 or value % 16 != 0:
                errors.append(f"{profile_name}: {dimension} must be a positive multiple of 16")
        if not isinstance(profile.get("steps"), int) or profile["steps"] <= 0:
            errors.append(f"{profile_name}: steps must be positive")
        if profile.get("sampling_method") not in {"euler", "euler_a", "heun", "dpm++2m"}:
            errors.append(f"{profile_name}: unsupported sampling method")

    return errors


def manifest_asset_path(asset: dict[str, Any]) -> Path:
    return resolve_under(ARTIFACTS_ROOT, asset["relative_path"])


def runtime_archive_path(asset: dict[str, Any]) -> Path:
    return ARTIFACTS_ROOT / "downloads" / asset["asset_name"]


def runtime_install_root(manifest: dict[str, Any]) -> Path:
    return ARTIFACTS_ROOT / "runtime" / manifest["runtime"]["id"]


def find_runtime_executable(manifest: dict[str, Any]) -> Path:
    install_root = runtime_install_root(manifest)
    candidates = sorted(install_root.rglob("sd-cli.exe")) if install_root.exists() else []
    if not candidates:
        raise LabError("sd-cli.exe is not installed; run install-runtime first")
    return candidates[0]


def download_with_curl(url: str, destination: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise LabError("curl is required for resumable downloads")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    command = [
        curl,
        "--location",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    print(f"Downloading {destination.name}")
    print(f"  partial: {partial}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise LabError(
            f"Download failed with exit code {completed.returncode}; "
            f"the partial file was retained for resume"
        )
    os.replace(partial, destination)


def ensure_download(
    *,
    name: str,
    url: str,
    destination: Path,
    size: int,
    sha256: str,
) -> None:
    valid, reason = validate_file(destination, size, sha256, hash_file=True)
    if valid:
        print(f"{name}: already verified")
        return
    if destination.exists():
        quarantine = destination.with_name(
            destination.name + f".invalid-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        print(f"{name}: {reason}; preserving invalid file as {quarantine.name}")
        destination.replace(quarantine)
    download_with_curl(url, destination)
    valid, reason = validate_file(destination, size, sha256, hash_file=True)
    if not valid:
        raise LabError(f"{name}: downloaded file failed verification: {reason}")
    print(f"{name}: SHA-256 verified")


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise LabError(f"Unsafe ZIP member: {member.filename}") from exc
        bundle.extractall(destination)


def install_runtime(manifest: dict[str, Any]) -> None:
    runtime = manifest["runtime"]
    archive = runtime_archive_path(runtime)
    ensure_download(
        name=runtime["id"],
        url=runtime["url"],
        destination=archive,
        size=runtime["size"],
        sha256=runtime["sha256"],
    )
    cuda_runtime = runtime["cuda_runtime"]
    cuda_archive = runtime_archive_path(cuda_runtime)
    ensure_download(
        name=f"{runtime['id']}-cuda-runtime",
        url=cuda_runtime["url"],
        destination=cuda_archive,
        size=cuda_runtime["size"],
        sha256=cuda_runtime["sha256"],
    )
    install_root = runtime_install_root(manifest)
    if not (install_root.exists() and list(install_root.rglob("sd-cli.exe"))):
        temporary = install_root.with_name(install_root.name + ".extracting")
        if temporary.exists():
            shutil.rmtree(temporary)
        safe_extract_zip(archive, temporary)
        if not list(temporary.rglob("sd-cli.exe")):
            raise LabError("Runtime archive did not contain sd-cli.exe")
        install_root.parent.mkdir(parents=True, exist_ok=True)
        if install_root.exists():
            shutil.rmtree(install_root)
        temporary.replace(install_root)
    cuda_marker = install_root / f".cuda-runtime-{cuda_runtime['sha256']}.installed"
    if not cuda_marker.is_file():
        safe_extract_zip(cuda_archive, install_root)
        cuda_marker.write_text(
            json.dumps(
                {
                    "asset": cuda_runtime["asset_name"],
                    "sha256": cuda_runtime["sha256"],
                    "installed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Installed runtime: {find_runtime_executable(manifest)}")


def bundle_asset_ids(manifest: dict[str, Any], bundle_name: str) -> list[str]:
    try:
        bundle = manifest["bundles"][bundle_name]
    except KeyError as exc:
        raise LabError(f"Unknown bundle: {bundle_name}") from exc
    return [bundle["transformer"], bundle["text_encoder"], bundle["vae"]]


def download_models(manifest: dict[str, Any], bundle_name: str) -> None:
    for asset_id in bundle_asset_ids(manifest, bundle_name):
        asset = manifest["assets"][asset_id]
        ensure_download(
            name=asset_id,
            url=asset["url"],
            destination=manifest_asset_path(asset),
            size=asset["size"],
            sha256=asset["sha256"],
        )


def get_prompt(prompt_id: str) -> dict[str, Any]:
    prompts = load_json(PROMPTS_PATH)["prompts"]
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    raise LabError(f"Unknown prompt id: {prompt_id}")


def bundle_paths(manifest: dict[str, Any], bundle_name: str) -> dict[str, Path]:
    bundle = manifest["bundles"][bundle_name]
    return {
        key: manifest_asset_path(manifest["assets"][bundle[key]])
        for key in ("transformer", "text_encoder", "vae")
    }


def build_sd_arguments(
    manifest: dict[str, Any],
    *,
    executable: Path,
    bundle_name: str,
    profile_name: str,
    prompt: str,
    seed: int,
    output: Path,
) -> list[str]:
    try:
        profile = manifest["profiles"][profile_name]
    except KeyError as exc:
        raise LabError(f"Unknown profile: {profile_name}") from exc
    paths = bundle_paths(manifest, bundle_name)
    arguments = [
        str(executable),
        "--diffusion-model",
        str(paths["transformer"]),
        "--vae",
        str(paths["vae"]),
        "--llm",
        str(paths["text_encoder"]),
        "-p",
        prompt,
        "--cfg-scale",
        str(profile["cfg_scale"]),
        "--sampling-method",
        str(profile["sampling_method"]),
        "--flow-shift",
        str(profile["flow_shift"]),
        "-W",
        str(profile["width"]),
        "-H",
        str(profile["height"]),
        "--steps",
        str(profile["steps"]),
        "--seed",
        str(seed),
        "--rng",
        "cuda",
        "--output",
        str(output),
    ]
    if profile.get("offload_to_cpu"):
        arguments.append("--offload-to-cpu")
    if profile.get("vae_on_cpu"):
        arguments.extend(["--backend", "vae=cpu"])
    if profile.get("auto_fit"):
        arguments.append("--auto-fit")
    if profile.get("diffusion_flash_attention"):
        arguments.append("--diffusion-fa")
    if profile.get("max_vram_gib") is not None:
        arguments.extend(["--max-vram", str(profile["max_vram_gib"])])
    if profile.get("verbose"):
        arguments.append("--verbose")
    return arguments


def redact_arguments(arguments: Iterable[str]) -> list[str]:
    return list(arguments)


def sample_gpu() -> dict[str, Any] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    command = [
        nvidia_smi,
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    fields = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        return None
    return {
        "timestamp": fields[0],
        "memory_used_mib": float(fields[1]),
        "memory_total_mib": float(fields[2]),
        "gpu_utilization_percent": float(fields[3]),
        "temperature_c": float(fields[4]),
        "power_w": float(fields[5]),
    }


def probe_runtime_cuda(manifest: dict[str, Any]) -> tuple[bool, str]:
    try:
        executable = find_runtime_executable(manifest)
    except LabError as exc:
        return False, str(exc)
    completed = subprocess.run(
        [str(executable), "--list-devices"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    cuda_lines = [line.strip() for line in output.splitlines() if line.startswith("CUDA")]
    if not cuda_lines:
        return False, "runtime did not enumerate a CUDA device"
    return True, "; ".join(cuda_lines)


def telemetry_worker(stop: threading.Event, samples: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        sample = sample_gpu()
        if sample:
            samples.append(sample)
        stop.wait(1.0)


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise LabError(f"Output is not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def normalize_ocr_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def best_ocr_phrase_score(expected: str, variants: dict[str, str]) -> float:
    expected_normalized = normalize_ocr_text(expected)
    expected_tokens = expected_normalized.split()
    best = 0.0
    for value in variants.values():
        tokens = normalize_ocr_text(value).split()
        for size in range(max(1, len(expected_tokens) - 1), len(expected_tokens) + 2):
            for start in range(0, max(0, len(tokens) - size + 1)):
                candidate = " ".join(tokens[start : start + size])
                score = difflib.SequenceMatcher(
                    None, expected_normalized, candidate
                ).ratio()
                best = max(best, score)
    return round(best, 3)


def run_scene_ocr(path: Path, tesseract: str) -> dict[str, str]:
    """OCR a scene through overlapping crops to reduce whole-image false negatives."""
    from PIL import Image, ImageOps

    outputs: dict[str, str] = {}
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        regions = {
            "full": rgb,
            "top": rgb.crop((0, 0, width, int(height * 0.5))),
            "bottom": rgb.crop((0, int(height * 0.4), width, height)),
            "bottom-left": rgb.crop(
                (0, int(height * 0.4), int(width * 0.65), height)
            ),
            "centre": rgb.crop(
                (
                    int(width * 0.1),
                    int(height * 0.1),
                    int(width * 0.9),
                    int(height * 0.9),
                )
            ),
        }
        for region_name, region in regions.items():
            enhanced = ImageOps.autocontrast(ImageOps.grayscale(region))
            enhanced = enhanced.resize(
                (enhanced.width * 2, enhanced.height * 2),
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            enhanced.save(buffer, format="PNG")
            for psm in (6, 11):
                completed = subprocess.run(
                    [tesseract, "stdin", "stdout", "--psm", str(psm)],
                    input=buffer.getvalue(),
                    capture_output=True,
                    check=False,
                )
                text_output = completed.stdout.decode("utf-8", errors="replace").strip()
                if text_output:
                    outputs[f"{region_name}-psm{psm}"] = text_output
    return outputs


def analyze_image(path: Path, prompt: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {"file_size_bytes": path.stat().st_size}
    try:
        from PIL import Image, ImageFilter, ImageStat

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            luma = rgb.convert("L")
            luma_stat = ImageStat.Stat(luma)
            edges = luma.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            histogram = luma.histogram()
            pixel_count = max(1, luma.width * luma.height)
            analysis.update(
                {
                    "luma_mean": round(luma_stat.mean[0], 3),
                    "luma_stddev": round(luma_stat.stddev[0], 3),
                    "entropy": round(luma.entropy(), 3),
                    "edge_variance": round(edge_stat.var[0], 3),
                    "clipped_black_percent": round(
                        100 * sum(histogram[:3]) / pixel_count, 4
                    ),
                    "clipped_white_percent": round(
                        100 * sum(histogram[-3:]) / pixel_count, 4
                    ),
                }
            )
    except (ImportError, OSError) as exc:
        analysis["pixel_metrics_error"] = str(exc)

    expected_text = prompt.get("expected_text")
    tesseract = shutil.which("tesseract")
    if expected_text and tesseract:
        ocr_variants = run_scene_ocr(path, tesseract)
        normalized_output = normalize_ocr_text("\n".join(ocr_variants.values()))
        expected_normalized = [normalize_ocr_text(item) for item in expected_text]
        matches = [item in normalized_output for item in expected_normalized]
        fuzzy_scores = [
            best_ocr_phrase_score(item, ocr_variants) for item in expected_text
        ]
        analysis.update(
            {
                "ocr_variants": ocr_variants,
                "ocr_expected": expected_text,
                "ocr_exact_phrase_matches": matches,
                "ocr_phrase_accuracy": round(sum(matches) / len(matches), 3),
                "ocr_fuzzy_phrase_scores": fuzzy_scores,
                "ocr_mean_fuzzy_score": round(
                    sum(fuzzy_scores) / len(fuzzy_scores), 3
                ),
            }
        )
    return analysis


def system_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
    }
    sample = sample_gpu()
    if sample:
        snapshot["gpu"] = sample
    return snapshot


def write_telemetry(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    return {
        "peak_vram_mib": max(sample["memory_used_mib"] for sample in samples),
        "peak_temperature_c": max(sample["temperature_c"] for sample in samples),
        "peak_power_w": max(sample["power_w"] for sample in samples),
        "mean_gpu_utilization_percent": round(
            sum(sample["gpu_utilization_percent"] for sample in samples)
            / len(samples),
            3,
        ),
        "telemetry_samples": len(samples),
    }


def new_run_directory(profile_name: str, prompt_id: str, seed: int) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS_ROOT / f"{timestamp}-{profile_name}-{prompt_id}-seed{seed}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_generation(
    manifest: dict[str, Any],
    *,
    bundle_name: str,
    profile_name: str,
    prompt_id: str,
    seed: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    prompt = get_prompt(prompt_id)
    profile = manifest["profiles"].get(profile_name)
    if not profile:
        raise LabError(f"Unknown profile: {profile_name}")
    selected_seed = profile["seed"] if seed is None else seed

    if dry_run:
        executable = Path("sd-cli.exe")
        output = Path("output.png")
    else:
        executable = find_runtime_executable(manifest)
        output = Path("pending.png")

    run_directory = None if dry_run else new_run_directory(profile_name, prompt_id, selected_seed)
    if run_directory:
        output = run_directory / "output.png"
    arguments = build_sd_arguments(
        manifest,
        executable=executable,
        bundle_name=bundle_name,
        profile_name=profile_name,
        prompt=prompt["prompt"],
        seed=selected_seed,
        output=output,
    )

    request = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bundle": bundle_name,
        "profile": profile_name,
        "prompt": prompt,
        "seed": selected_seed,
        "arguments": redact_arguments(arguments),
        "system": system_snapshot(),
    }

    if dry_run:
        print(json.dumps(request, indent=2))
        return {"status": "dry-run", **request}

    assert run_directory is not None
    for path in bundle_paths(manifest, bundle_name).values():
        if not path.is_file():
            raise LabError(f"Missing model component: {path}; run download-models first")
    (run_directory / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )

    log_path = run_directory / "process.log"
    telemetry: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=telemetry_worker, args=(stop, telemetry), daemon=True
    )
    print(f"Starting {profile_name}: {prompt_id}, seed {selected_seed}")
    print(f"Output directory: {run_directory}")
    started = time.perf_counter()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
        monitor.start()
        try:
            return_code = process.wait(timeout=profile["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
        finally:
            stop.set()
            monitor.join(timeout=5)
    elapsed = time.perf_counter() - started
    write_telemetry(run_directory / "gpu-telemetry.csv", telemetry)

    result: dict[str, Any] = {
        "status": "success" if return_code == 0 and output.is_file() else "failed",
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "output": str(output),
        "log": str(log_path),
        "run_directory": str(run_directory),
    }
    result.update(summarize_telemetry(telemetry))
    if output.is_file():
        width, height = png_dimensions(output)
        result["output_width"] = width
        result["output_height"] = height
        result["output_sha256"] = sha256_file(output)
        result["image_analysis"] = analyze_image(output, prompt)
        if width != profile["width"] or height != profile["height"]:
            result["status"] = "failed"
            result["error"] = (
                f"Output dimensions {width}x{height} do not match "
                f"{profile['width']}x{profile['height']}"
            )
    elif not timed_out:
        result["error"] = "Runner exited without producing output.png"

    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        raise LabError(f"Generation failed; inspect {log_path}")
    return result


def doctor(manifest: dict[str, Any], bundle_name: str, full_hash: bool) -> bool:
    checks: list[tuple[str, bool, str]] = []
    errors = validate_manifest(manifest)
    checks.append(("manifest", not errors, "; ".join(errors) if errors else "valid"))

    try:
        executable = find_runtime_executable(manifest)
        checks.append(("runtime", True, str(executable)))
    except LabError as exc:
        checks.append(("runtime", False, str(exc)))
    runtime_cuda, runtime_cuda_detail = probe_runtime_cuda(manifest)
    checks.append(("runtime CUDA backend", runtime_cuda, runtime_cuda_detail))

    for asset_id in bundle_asset_ids(manifest, bundle_name):
        asset = manifest["assets"][asset_id]
        valid, reason = validate_file(
            manifest_asset_path(asset),
            asset["size"],
            asset["sha256"],
            hash_file=full_hash,
        )
        checks.append((asset_id, valid, reason))

    gpu = sample_gpu()
    checks.append(("NVIDIA GPU", gpu is not None, json.dumps(gpu) if gpu else "unavailable"))
    free_bytes = shutil.disk_usage(LAB_ROOT).free
    checks.append(("free disk", free_bytes >= 10 * 1024**3, f"{free_bytes / 1024**3:.1f} GiB"))

    for name, valid, detail in checks:
        print(f"[{'PASS' if valid else 'FAIL'}] {name}: {detail}")
    return all(valid for _, valid, _ in checks)


def run_benchmark(
    manifest: dict[str, Any],
    *,
    bundle_name: str,
    profile_name: str,
    suite_name: str,
    dry_run: bool,
) -> None:
    data = load_json(PROMPTS_PATH)
    try:
        suite = data["suites"][suite_name]
    except KeyError as exc:
        raise LabError(f"Unknown benchmark suite: {suite_name}") from exc
    for prompt_id in suite["prompt_ids"]:
        for seed in suite["seeds"]:
            run_generation(
                manifest,
                bundle_name=bundle_name,
                profile_name=profile_name,
                prompt_id=prompt_id,
                seed=seed,
                dry_run=dry_run,
            )


def collect_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not RESULTS_ROOT.exists():
        return results
    for result_path in sorted(RESULTS_ROOT.glob("*/result.json")):
        try:
            result = load_json(result_path)
            request = load_json(result_path.parent / "request.json")
        except (OSError, json.JSONDecodeError):
            continue
        results.append({"result": result, "request": request, "path": result_path.parent})
    return results


def reanalyze_results() -> int:
    count = 0
    for item in collect_results():
        output = Path(item["result"]["output"])
        if not output.is_file():
            continue
        item["result"]["image_analysis"] = analyze_image(
            output, item["request"]["prompt"]
        )
        telemetry_path = item["path"] / "gpu-telemetry.csv"
        if telemetry_path.is_file():
            with telemetry_path.open("r", encoding="utf-8", newline="") as handle:
                telemetry = []
                for row in csv.DictReader(handle):
                    telemetry.append(
                        {
                            "memory_used_mib": float(row["memory_used_mib"]),
                            "temperature_c": float(row["temperature_c"]),
                            "power_w": float(row["power_w"]),
                            "gpu_utilization_percent": float(
                                row["gpu_utilization_percent"]
                            ),
                        }
                    )
            item["result"].update(summarize_telemetry(telemetry))
        (item["path"] / "result.json").write_text(
            json.dumps(item["result"], indent=2), encoding="utf-8"
        )
        count += 1
    print(f"Reanalyzed {count} completed output(s)")
    return count


def generate_report() -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "latest.md"
    results = collect_results()
    successful = [item for item in results if item["result"]["status"] == "success"]
    failed = [item for item in results if item["result"]["status"] != "success"]
    lines = [
        "# Image Generation Lab Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Completed runs: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(failed)}",
        "",
    ]
    if successful:
        lines.extend(
            [
                "| Profile | Prompt | Seed | Seconds | VRAM MiB | Temp C | Output |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in successful:
            result = item["result"]
            request = item["request"]
            output = Path(result["output"])
            try:
                relative_output = output.relative_to(REPORTS_ROOT)
            except ValueError:
                relative_output = Path(os.path.relpath(output, REPORTS_ROOT))
            lines.append(
                f"| {request['profile']} | {request['prompt']['id']} | "
                f"{request['seed']} | {result['elapsed_seconds']:.1f} | "
                f"{result.get('peak_vram_mib', 'n/a')} | "
                f"{result.get('peak_temperature_c', 'n/a')} | "
                f"[image]({relative_output.as_posix()}) |"
            )
        times = [item["result"]["elapsed_seconds"] for item in successful]
        peaks = [
            item["result"].get("peak_vram_mib")
            for item in successful
            if item["result"].get("peak_vram_mib") is not None
        ]
        temperatures = [
            item["result"].get("peak_temperature_c")
            for item in successful
            if item["result"].get("peak_temperature_c") is not None
        ]
        lines.extend(
            [
                "",
                f"Mean generation time: {sum(times) / len(times):.1f} seconds.",
                (
                    f"Highest observed VRAM: {max(peaks):.0f} MiB."
                    if peaks
                    else "VRAM telemetry was unavailable."
                ),
                (
                    f"Highest observed GPU temperature: {max(temperatures):.0f} C."
                    if temperatures
                    else "Temperature telemetry was unavailable."
                ),
            ]
        )
        text_runs = [
            item
            for item in successful
            if "ocr_phrase_accuracy"
            in item["result"].get("image_analysis", {})
        ]
        if text_runs:
            mean_ocr = sum(
                item["result"]["image_analysis"]["ocr_phrase_accuracy"]
                for item in text_runs
            ) / len(text_runs)
            fuzzy_runs = [
                item["result"]["image_analysis"].get("ocr_mean_fuzzy_score")
                for item in text_runs
                if item["result"]["image_analysis"].get("ocr_mean_fuzzy_score")
                is not None
            ]
            lines.append(
                f"Conservative automated exact-phrase OCR recall: {mean_ocr:.1%}."
            )
            if fuzzy_runs:
                lines.append(
                    f"Mean fuzzy OCR similarity: {sum(fuzzy_runs) / len(fuzzy_runs):.1%}."
                )
            lines.append(
                "Scene OCR can produce false negatives; text outputs require visual review."
            )
    if failed:
        lines.extend(["", "## Failures", ""])
        for item in failed:
            lines.append(
                f"- `{item['path'].name}`: {item['result'].get('error', 'runner failure')}"
            )
    if not results:
        lines.append("No completed generation results were found.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate pinned lab configuration")
    subparsers.add_parser("install-runtime", help="Install the pinned CUDA runtime")

    download = subparsers.add_parser("download-models", help="Download a model bundle")
    download.add_argument("--bundle", default="qwen-2512-q6-quality")

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime, models and GPU")
    doctor_parser.add_argument("--bundle", default="qwen-2512-q6-quality")
    doctor_parser.add_argument("--full-hash", action="store_true")

    generate = subparsers.add_parser("generate", help="Run one generation")
    generate.add_argument("--bundle", default="qwen-2512-q6-quality")
    generate.add_argument("--profile", default="smoke")
    generate.add_argument("--prompt-id", default="surfboard-cat")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--dry-run", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run a prompt suite")
    benchmark.add_argument("--bundle", default="qwen-2512-q6-quality")
    benchmark.add_argument("--profile", default="quality-1024")
    benchmark.add_argument("--suite", default="quality-core")
    benchmark.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("report", help="Generate a Markdown results report")
    subparsers.add_parser(
        "reanalyze", help="Recompute image and OCR metrics for completed runs"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(MANIFEST_PATH)
    errors = validate_manifest(manifest)
    if errors and args.command != "validate":
        raise LabError("Manifest is invalid: " + "; ".join(errors))

    if args.command == "validate":
        if errors:
            for error in errors:
                print(f"[FAIL] {error}")
            return 1
        print("[PASS] manifest and prompt configuration are valid")
        return 0
    if args.command == "install-runtime":
        install_runtime(manifest)
    elif args.command == "download-models":
        download_models(manifest, args.bundle)
    elif args.command == "doctor":
        return 0 if doctor(manifest, args.bundle, args.full_hash) else 1
    elif args.command == "generate":
        run_generation(
            manifest,
            bundle_name=args.bundle,
            profile_name=args.profile,
            prompt_id=args.prompt_id,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    elif args.command == "benchmark":
        run_benchmark(
            manifest,
            bundle_name=args.bundle,
            profile_name=args.profile,
            suite_name=args.suite,
            dry_run=args.dry_run,
        )
    elif args.command == "report":
        generate_report()
    elif args.command == "reanalyze":
        reanalyze_results()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LabError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
