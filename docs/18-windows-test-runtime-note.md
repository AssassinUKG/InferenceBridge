# Windows cargo test runtime note

## Symptom

`cargo test --no-run` succeeds, but `cargo test` fails when Windows starts the compiled test executable:

```text
STATUS_ENTRYPOINT_NOT_FOUND
0xc0000139
```

## Meaning

This is a runtime loader failure, not a Rust compile failure. The test binary was built, but Windows loaded a DLL that does not export a symbol the binary or one of its native dependencies expects.

Common causes:

- an older native DLL appears earlier on `PATH`
- CUDA/Vulkan/runtime DLLs are mixed across releases
- a DLL from another local LLM/runtime install shadows the expected one
- the test process sees a different DLL search path than the release app

## Current impact

- `cargo build` passes.
- `cargo test --no-run` passes.
- `cargo clippy` runs.
- The release app builds and runs.
- The compiled test binary cannot be launched in this Windows environment until the DLL mismatch is isolated.

## Suggested investigation

1. Run the failing test executable directly from `src-tauri/target/debug/deps` and note the exact Windows loader dialog or error.
2. Inspect loaded dependency resolution with a tool such as Dependencies.exe.
3. Compare `PATH` entries for llama.cpp, CUDA, Vulkan, Tauri/WebView, and other local inference tools.
4. Temporarily run with a minimal `PATH` containing only Windows system dirs, Rust toolchain dirs, and the repo/runtime dirs required by the test.
5. If the missing entrypoint names a CUDA/Vulkan DLL, align the managed llama.cpp runtime DLLs with the active driver/toolkit version.

Until this is fixed, use `cargo test --no-run` for compile coverage and rely on API smoke tests plus release builds for runtime confidence.
