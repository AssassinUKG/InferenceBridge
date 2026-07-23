import assert from "node:assert/strict";
import test from "node:test";

import { isClearableDownload, mergeDownloadSnapshots } from "../src/lib/downloads.ts";

const download = (id, downloadedBytes, status = "Downloading") => ({
  id,
  filename: `${id}.gguf`,
  downloaded_bytes: downloadedBytes,
  status,
});

test("restored downloads populate an empty Download Manager", () => {
  const restored = [download("model-a", 4096, "Resuming")];

  assert.deepEqual(mergeDownloadSnapshots(restored, {}), {
    "model-a": restored[0],
  });
});

test("live progress wins when it arrives before the startup snapshot", () => {
  const stale = download("model-a", 4096, "Interrupted");
  const live = download("model-a", 8192, "Downloading");

  assert.equal(
    mergeDownloadSnapshots([stale], { "model-a": live })["model-a"],
    live,
  );
});

test("startup restore keeps unrelated live and durable downloads", () => {
  const restored = download("model-a", 4096, "Paused");
  const live = download("model-b", 8192);

  assert.deepEqual(
    Object.keys(mergeDownloadSnapshots([restored], { "model-b": live })).sort(),
    ["model-a", "model-b"],
  );
});

test("Clear Done keeps resumable failures until the user resumes or discards them", () => {
  assert.equal(isClearableDownload({ done: true, resumable: false, status: "Completed" }), true);
  assert.equal(isClearableDownload({ done: true, resumable: true, status: "Failed" }), false);
  assert.equal(isClearableDownload({ done: false, resumable: true, status: "Downloading" }), false);
  assert.equal(isClearableDownload({ done: true, resumable: false, status: "Cleanup pending" }), false);
});
