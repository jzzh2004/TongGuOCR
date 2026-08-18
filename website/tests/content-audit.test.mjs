import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const content = await readFile(new URL("src/content/project.ts", root), "utf8");
const app = await readFile(new URL("src/App.tsx", root), "utf8");
const html = await readFile(new URL("dist/index.html", root), "utf8");

test("includes every paper figure and table exactly once in content data", async () => {
  for (let number = 1; number <= 4; number += 1) {
    const matches = content.match(new RegExp(`id: \\"figure-${number}\\"`, "g")) ?? [];
    assert.equal(matches.length, 1, `Figure ${number} must be declared once`);
    await access(new URL(`public/paper-assets/figures/figure-${number}.${number === 1 || number === 4 ? "webp" : "png"}`, root));
  }

  for (let number = 1; number <= 9; number += 1) {
    const matches = content.match(new RegExp(`id: \\"table-${number}\\"`, "g")) ?? [];
    assert.equal(matches.length, 1, `Table ${number} must be declared once`);
    await access(new URL(`public/paper-assets/tables/table-${number}.png`, root));
  }
});

test("keeps project links centrally configurable", () => {
  assert.match(content, /label: "Paper", url: "https:\/\/arxiv\.org\/abs\/2608\.07917"/);
  assert.match(content, /label: "Demo", url: "https:\/\/guji-ocr\.com"/);
  assert.match(content, /label: "GitHub", url: "https:\/\/github\.com\/jzzh2004\/TongGuOCR"/);
  assert.match(content, /label: "Hugging Face", url: null/);
  assert.doesNotMatch(app, /disabled title=/);
  assert.match(app, /className="resource-link is-unavailable"/);
  assert.match(app, /target="_blank" rel="noreferrer noopener"/);
});

test("preserves the complete nine-author paper byline", () => {
  const expectedAuthors = [
    "Zhongheng Zhou",
    "Yi Sun",
    "Huiguo He",
    "Yuyi Zhang",
    "Peirong Zhang",
    "Yulin Fang",
    "Dezhi Peng",
    "Minghui Liao",
    "Lianwen Jin",
  ];
  for (const author of expectedAuthors) assert.match(content, new RegExp(`name: \\"${author}\\"`));
  assert.equal((content.match(/name: "/g) ?? []).length, 9);
});

test("publishes paper metadata in the production HTML", () => {
  assert.match(html, /TongGuOCR/);
  assert.match(html, /Chinese Historical Document OCR/);
  assert.match(html, /og:image/);
});
