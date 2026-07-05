/**
 * Convert all single-page PDF figures under assets/paper_note/ to PNG images
 * in public/paper-node/ so they can be referenced as inline images in Markdown.
 *
 * Uses MuPDF WASM for high-quality rendering (embedded fonts, vector graphics).
 *
 * Usage:  node scripts/pdf-to-png.js          — incremental convert
 *         node scripts/pdf-to-png.js --force  — re-convert everything
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as mupdf from "mupdf";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const ASSETS_DIR = path.join(ROOT, "assets", "paper_note");
const PUBLIC_DIR = path.join(ROOT, "public", "paper-node");

const SCALE = 3;
const FORCE = process.argv.includes("--force");

function collectPdfs(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectPdfs(full));
    } else if (entry.name.toLowerCase().endsWith(".pdf")) {
      results.push(full);
    }
  }
  return results;
}

function assetPathToPublicPath(assetPath) {
  const rel = path.relative(ASSETS_DIR, assetPath);
  const parts = rel.split(path.sep);
  parts[parts.length - 1] = parts[parts.length - 1].replace(/\.pdf$/i, ".png");
  return path.join(PUBLIC_DIR, ...parts);
}

function needsConvert(pdfPath, pngPath) {
  if (FORCE) return true;
  if (!fs.existsSync(pngPath)) return true;
  return fs.statSync(pdfPath).mtimeMs > fs.statSync(pngPath).mtimeMs;
}

function convertOne(pdfPath, pngPath) {
  fs.mkdirSync(path.dirname(pngPath), { recursive: true });

  const data = fs.readFileSync(pdfPath);
  const doc = mupdf.Document.openDocument(data, "application/pdf");
  const page = doc.loadPage(0);

  const matrix = mupdf.Matrix.scale(SCALE, SCALE);
  const pixmap = page.toPixmap(matrix, mupdf.ColorSpace.DeviceRGB, false, true);
  const pngBuf = pixmap.asPNG();
  fs.writeFileSync(pngPath, pngBuf);
}

function main() {
  const pdfs = collectPdfs(ASSETS_DIR);
  if (pdfs.length === 0) {
    console.log("No PDF files found under assets/paper_note/.");
    return;
  }

  console.log(`Found ${pdfs.length} PDF(s) under assets/paper_note/`);
  let converted = 0;
  let skipped = 0;
  let failed = 0;

  for (const pdf of pdfs) {
    const png = assetPathToPublicPath(pdf);
    const relPdf = path.relative(ROOT, pdf);
    const relPng = path.relative(ROOT, png);

    if (!needsConvert(pdf, png)) {
      skipped++;
      continue;
    }

    try {
      process.stdout.write(`  ${relPdf} → ${relPng} ... `);
      convertOne(pdf, png);
      converted++;
      console.log("OK");
    } catch (err) {
      failed++;
      console.error(`FAIL\n    ${err.message}`);
    }
  }

  console.log(
    `\nDone: ${converted} converted, ${skipped} skipped (up-to-date), ${failed} failed.`
  );
}

main();
