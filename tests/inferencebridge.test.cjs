const axios = require("axios");
const assert = require("node:assert/strict");

const baseUrl = process.env.INFBRIDGE_API_URL;

async function main() {
  if (!baseUrl) {
    console.log("Skipping API smoke test. Set INFBRIDGE_API_URL to a running server to enable it.");
    return;
  }

  const resp = await axios.get(`${baseUrl}/v1/models`);
  assert.equal(resp.status, 200);
  assert.ok(Object.hasOwn(resp.data, "data"));
  assert.ok(Array.isArray(resp.data.data));
  console.log(`API smoke test passed for ${baseUrl}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
