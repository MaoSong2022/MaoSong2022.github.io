/**
 * Astro integration: generate 1200x630 PNG OpenGraph cards at build time.
 *
 * Outputs dist/og/<slug>.png for each article bundle that does not specify
 * a custom seoThumbImage. Uses sharp (already a project dependency) to render
 * clean SVG-based social preview cards.
 */
import { promises as fs } from "node:fs";
import { resolve } from "node:path";

function stripQuotes(s) {
  return String(s || "")
    .replace(/^["']|["']$/g, "")
    .trim();
}

function escapeXml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function extractFrontmatter(content) {
  const m = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!m) return {};
  const fm = {};
  const lines = m[1].split(/\r?\n/);
  for (const line of lines) {
    const kv = line.match(/^([a-zA-Z_]\w*)\s*:\s*(.*)$/);
    if (kv) {
      fm[kv[1].trim()] = stripQuotes(kv[2]);
    }
  }
  return fm;
}

/**
 * Keep SVG text ASCII-safe so Linux CI runners without CJK fonts never render
 * missing-glyph tofu boxes. Falls back to a humanized slug if needed.
 */
function toSafeCardTitle(rawTitle, slug) {
  const flat = String(rawTitle || "")
    .replace(/<[^>]*>/g, "")
    .replace(/\\n/g, " ")
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const ascii = flat
    .replace(/[^\x20-\x7E]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^[:\-\s]+|[:\-\s]+$/g, "")
    .trim();
  if (ascii.length >= 6) return ascii;
  return slug
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function wrapLines(text, maxCharsPerLine = 28, maxLines = 3) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxCharsPerLine) {
      current = candidate;
    } else {
      if (current) lines.push(current);
      current = word;
      if (lines.length === maxLines - 1) break;
    }
  }
  if (current && lines.length < maxLines) {
    lines.push(current);
  }
  const joined = lines.join(" ");
  if (joined.length < text.length && lines.length > 0) {
    const last = lines[lines.length - 1];
    lines[lines.length - 1] =
      last.length > maxCharsPerLine - 3
        ? `${last.slice(0, maxCharsPerLine - 3)}...`
        : `${last}...`;
  }
  return lines;
}

function buildOgSvg({ title, slug, published }) {
  const lines = wrapLines(title, 26, 3);
  const fontSize = lines.length >= 3 ? 52 : 60;
  const lineHeight = Math.round(fontSize * 1.22);
  const startY = lines.length === 1 ? 315 : lines.length === 2 ? 280 : 250;

  const titleSpans = lines
    .map(
      (line, idx) =>
        `<text x="84" y="${startY + idx * lineHeight}" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="${fontSize}" font-weight="700" fill="#F8FAFC">${escapeXml(line)}</text>`,
    )
    .join("\n    ");

  const metaLine = published ? `${published.slice(0, 10)}  ·  /blog/${slug}/` : `/blog/${slug}/`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="55%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#0B1120"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <rect x="0" y="0" width="1200" height="8" fill="url(#accent)"/>
  <circle cx="1080" cy="110" r="220" fill="#38BDF8" fill-opacity="0.06"/>
  <circle cx="1020" cy="540" r="180" fill="#818CF8" fill-opacity="0.06"/>
  <rect x="84" y="96" width="210" height="38" rx="19" fill="#38BDF8" fill-opacity="0.14" stroke="#38BDF8" stroke-opacity="0.35"/>
  <text x="108" y="121" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="18" font-weight="600" fill="#7DD3FC" letter-spacing="1.2">RESEARCH NOTES</text>
  ${titleSpans}
  <line x1="84" y1="504" x2="1116" y2="504" stroke="#334155" stroke-width="1.5"/>
  <text x="84" y="556" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="24" font-weight="600" fill="#E2E8F0">Mao Song</text>
  <text x="236" y="556" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="22" fill="#94A3B8">maosong.website</text>
  <text x="1116" y="556" text-anchor="end" font-family="DejaVu Mono, Liberation Mono, monospace" font-size="19" fill="#64748B">${escapeXml(metaLine)}</text>
</svg>`;
}

export default function generateOgImages() {
  return {
    name: "generate-og-images",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        let sharp;
        try {
          const mod = await import("sharp");
          sharp = mod.default ?? mod;
        } catch (err) {
          console.warn("[og-images] sharp unavailable, skipping OG generation:", err?.message || err);
          return;
        }

        const contentRoot = resolve(process.cwd(), "src/content");
        const outDir = resolve(dir.pathname, "og");
        await fs.mkdir(outDir, { recursive: true });

        const entries = await fs.readdir(contentRoot, { withFileTypes: true });
        let generated = 0;

        for (const entry of entries) {
          if (!entry.isDirectory() || entry.name === "demo") continue;
          const slug = entry.name;
          let raw = "";
          for (const ext of ["article.mdx", "article.md"]) {
            try {
              raw = await fs.readFile(resolve(contentRoot, slug, ext), "utf-8");
              break;
            } catch {}
          }
          if (!raw) continue;

          const fm = extractFrontmatter(raw);
          if (fm.seoThumbImage) continue;

          const cardTitle = toSafeCardTitle(fm.title, slug);
          const svg = buildOgSvg({
            title: cardTitle,
            slug,
            published: fm.published,
          });

          try {
            await sharp(Buffer.from(svg))
              .png({ compressionLevel: 9 })
              .toFile(resolve(outDir, `${slug}.png`));
            generated += 1;
          } catch (err) {
            console.warn(`[og-images] failed for ${slug}:`, err?.message || err);
          }
        }

        console.log(`[og-images] Generated ${generated} OG card images in dist/og/`);
      },
    },
  };
}
