import assert from "node:assert/strict";
import test from "node:test";

import {
  HUB_DETAIL_MIN_WIDTH,
  HUB_LIST_MIN_WIDTH,
  HUB_SEPARATOR_WIDTH,
  adjustHubListWidthByKeyboard,
  chooseRecommendedQuant,
  clampHubListWidth,
  defaultHubListWidth,
  resizeHubListWidth,
} from "../src/lib/modelHubLayout.ts";

const quant = (name, extra = {}) => ({ quant: name, filename: `${name}.gguf`, ...extra });

test("a generic Q4_K is recommended ahead of Q6_K", () => {
  const q6 = quant("Q6_K");
  const q4 = quant("Q4_K");
  assert.equal(chooseRecommendedQuant([q6, q4]), q4);
});

test("Q4_K_M is the preferred balanced quant", () => {
  const q5 = quant("Q5_K_M");
  const q4 = quant("q4-k-m");
  const q4Generic = quant("Q4_K");
  assert.equal(chooseRecommendedQuant([q5, q4Generic, q4]), q4);
});

test("installation state does not influence the recommendation", () => {
  const installedQ4 = quant("Q4_K_M", { installed: true });
  const availableQ6 = quant("Q6_K", { installed: false });
  assert.equal(chooseRecommendedQuant([availableQ6, installedQ4]), installedQ4);
  assert.equal(chooseRecommendedQuant([]), null);
});

test("the default split is about 36 percent and respects panel minima", () => {
  assert.equal(defaultHubListWidth(1600), 576);
  assert.equal(defaultHubListWidth(900), 324);
  assert.equal(defaultHubListWidth(700), HUB_LIST_MIN_WIDTH);
});

test("list widths clamp between the list and detail minima", () => {
  const containerWidth = 1200;
  const maximum = containerWidth - HUB_DETAIL_MIN_WIDTH - HUB_SEPARATOR_WIDTH;

  assert.equal(clampHubListWidth(containerWidth, 100), HUB_LIST_MIN_WIDTH);
  assert.equal(clampHubListWidth(containerWidth, 900), maximum);
  assert.equal(clampHubListWidth(containerWidth, 480), 480);
  assert.equal(resizeHubListWidth(480, -1000, containerWidth), HUB_LIST_MIN_WIDTH);
  assert.equal(resizeHubListWidth(480, 1000, containerWidth), maximum);
});

test("keyboard resizing uses fluid steps and the same clamps", () => {
  const containerWidth = 1200;
  const maximum = containerWidth - HUB_DETAIL_MIN_WIDTH - HUB_SEPARATOR_WIDTH;

  assert.equal(adjustHubListWidthByKeyboard(480, "ArrowLeft", containerWidth), 456);
  assert.equal(adjustHubListWidthByKeyboard(480, "ArrowRight", containerWidth), 504);
  assert.equal(adjustHubListWidthByKeyboard(480, "ArrowRight", containerWidth, 7), 487);
  assert.equal(adjustHubListWidthByKeyboard(480, "Home", containerWidth), HUB_LIST_MIN_WIDTH);
  assert.equal(adjustHubListWidthByKeyboard(480, "End", containerWidth), maximum);
  assert.equal(adjustHubListWidthByKeyboard(480, "Escape", containerWidth), 480);
});
