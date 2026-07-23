import assert from "node:assert/strict";
import test from "node:test";

import {
  formatImageDuration,
  formatImageFileSize,
  formatImageSampler,
  imageAspectRatio,
  imageDataUrlByteSize,
  imageModelLabel,
  parseGeneratedImageMetadata,
} from "../src/lib/generatedImagePresentation.ts";

test("parses stored generated-image metadata safely", () => {
  assert.deepEqual(
    parseGeneratedImageMetadata('{"width":1024,"height":1024,"steps":50}'),
    { width: 1024, height: 1024, steps: 50 },
  );
  assert.equal(parseGeneratedImageMetadata("{broken"), null);
  assert.equal(parseGeneratedImageMetadata("[]"), null);
});

test("formats useful image-generation details", () => {
  assert.equal(formatImageDuration(211.2), "3m 31s");
  assert.equal(formatImageFileSize(2_315_165), "2.2 MB");
  assert.equal(imageAspectRatio(1664, 928), "16:9");
  assert.equal(formatImageSampler("dpm++2m"), "DPM++ 2M");
  assert.equal(
    imageModelLabel({ bundle_id: "qwen-image-2512-q6" }),
    "Qwen-Image 2512 · Q6",
  );
});

test("derives file size from a base64 data URL for older attachments", () => {
  assert.equal(imageDataUrlByteSize("data:image/png;base64,QUJDRA=="), 4);
  assert.equal(imageDataUrlByteSize("https://example.com/image.png"), null);
});
