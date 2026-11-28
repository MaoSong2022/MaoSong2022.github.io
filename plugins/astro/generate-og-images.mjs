/**
 * Astro integration: generate 1200x630 PNG OpenGraph cards at build time.
 *
 * Outputs dist/og/<slug>.png for each article bundle that does not specify
 * a custom seoThumbImage. Uses sharp (already a project dependency) to render
 * clean SVG-based social preview cards.
 */
import { promises as fs } from "node:fs";
import { resolve } from "node:path";

const OG_ENGLISH_TITLES = {
  rl4llm: "RL4LLM: Reinforcement Learning for LLMs (MDP, PPO & GRPO)",
  "llm-resource-computation": "LLM Resource Computation: Params, FLOPs & Memory",
  tokenizer: "Tokenizer: BPE, Unigram & Engineering Practice",
  "train-inference-mismatch": "Training-Inference Mismatch in LLM Reinforcement Learning",
  parallelism: "Distributed Parallel Training for LLMs: DP, TP, PP, CP & EP",
  attention: "Attention Mechanisms: MHA, GQA, MLA, Sparse & Linear",
  "policy-optimization": "LLM Policy Optimization: PPO, GRPO, DAPO & GSPO",
  inference: "LLM Inference Optimization: KV Cache, Batching & Speculative Decoding",
  "scaling-law": "Scaling Laws for Large Language Model Pre-training",
  deepseek: "DeepSeek Series: MoE, MLA & V3/V4 Technical Evolution",
  qwen: "Qwen Series: Architecture, Pre-training & Post-training",
  kimi: "Kimi Series: Long-CoT RL, MoE & Agentic Practice",
  "on-policy-distillation": "On-Policy Distillation (OPD): Principles & Progress",
  "agentic-rl": "Agentic RL: POMDP, Infrastructure & Algorithms",
  "gpu-tpu": "GPU vs TPU Architecture for Large-Scale LLM Training",
  "kat-coder": "KAT-Coder Series: Multi-Expert Agentic Coding Models",
  apertus: "Apertus Series: Open Multilingual & Multimodal LLMs",
  mai: "MAI Series: Large-Scale MoE Pre-training at Microsoft AI",
  "tech-reports-2026-q4": "Tech Reports Review (2026 Q4)",
};

const CONTENT_TYPE_BADGES = {
  analysis: { label: "DEEP DIVE", color: "#A78BFA", bg: "#8B5CF6" },
  learning: { label: "LEARNING NOTES", color: "#7DD3FC", bg: "#38BDF8" },
  topics: { label: "TOPIC OVERVIEW", color: "#6EE7B7", bg: "#10B981" },
  reports: { label: "TECHNICAL REPORT", color: "#FCD34D", bg: "#F59E0B" },
};

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

function formatCardDate(raw) {
  if (!raw) return "";
  const trimmed = String(raw).trim();
  const isoMatch = trimmed.match(/^(\d{4}-\d{2}-\d{2})/);
  if (isoMatch) return isoMatch[1];
  const parsed = new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) return "";
  const y = parsed.getFullYear();
  const m = String(parsed.getMonth() + 1).padStart(2, "0");
  const d = String(parsed.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

async function readEditorialContentTypes(cwd) {
  const map = new Map();
  let raw = "";
  try {
    raw = await fs.readFile(resolve(cwd, "src/data/article-editorial.ts"), "utf-8");
  } catch {
    return map;
  }
  const re = /^\s*(?:["']([^"']+)["']|([a-zA-Z0-9_-]+))\s*:\s*(.+?),\s*$/gm;
  let m;
  while ((m = re.exec(raw)) !== null) {
    const slug = m[1] || m[2];
    const rhs = m[3];
    if (rhs.includes("report(") || rhs.includes('"reports"')) map.set(slug, "reports");
    else if (rhs.includes("topic(") || rhs.includes('"topics"')) map.set(slug, "topics");
    else if (rhs.includes('"analysis"')) map.set(slug, "analysis");
    else if (rhs.includes("learning(") || rhs.includes('"learning"')) map.set(slug, "learning");
  }
  return map;
}

/**
 * Keep SVG text ASCII-safe so Linux CI runners without CJK fonts never render
 * missing-glyph tofu boxes, while avoiding fragmented acronyms from stripping
 * Chinese characters out of bilingual titles.
 */
function toSafeCardTitle(fm, slug) {
  if (fm.ogTitle && !/[^\x20-\x7E]/.test(fm.ogTitle)) {
    return fm.ogTitle.trim();
  }
  if (OG_ENGLISH_TITLES[slug]) {
    return OG_ENGLISH_TITLES[slug];
  }
  const flat = String(fm.title || "")
    .replace(/<[^>]*>/g, "")
    .replace(/\\n/g, " ")
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  // Pure ASCII title can be used directly
  if (flat.length >= 4 && !/[^\x20-\x7E]/.test(flat)) {
    return flat;
  }

  // For bilingual titles like "Topic Name：中文副标题", use the clean ASCII prefix
  const prefix = flat.split(/[：:]/)[0]?.trim() || "";
  if (prefix.length >= 4 && !/[^\x20-\x7E]/.test(prefix)) {
    return prefix;
  }

  // Fallback to a cleanly humanized slug
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

function buildOgSvg({ title, slug, published, contentType }) {
  const lines = wrapLines(title, 28, 3);
  const fontSize = lines.length >= 3 ? 50 : 58;
  const lineHeight = Math.round(fontSize * 1.24);
  const startY = lines.length === 1 ? 315 : lines.length === 2 ? 280 : 248;

  const badge = CONTENT_TYPE_BADGES[contentType] ?? {
    label: "RESEARCH NOTES",
    color: "#7DD3FC",
    bg: "#38BDF8",
  };
  const badgeWidth = Math.max(180, badge.label.length * 13 + 56);

  const titleSpans = lines
    .map(
      (line, idx) =>
        `<text x="84" y="${startY + idx * lineHeight}" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="${fontSize}" font-weight="700" fill="#F8FAFC">${escapeXml(line)}</text>`,
    )
    .join("\n    ");

  const formattedDate = formatCardDate(published);
  const metaLine = formattedDate
    ? `${formattedDate}  ·  /blog/${slug}/`
    : `/blog/${slug}/`;

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
  <rect x="84" y="96" width="${badgeWidth}" height="38" rx="19" fill="${badge.bg}" fill-opacity="0.14" stroke="${badge.bg}" stroke-opacity="0.35"/>
  <text x="112" y="121" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif" font-size="18" font-weight="600" fill="${badge.color}" letter-spacing="1.2">${escapeXml(badge.label)}</text>
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

        const cwd = process.cwd();
        const contentRoot = resolve(cwd, "src/content");
        const outDir = resolve(dir.pathname, "og");
        await fs.mkdir(outDir, { recursive: true });
        const contentTypes = await readEditorialContentTypes(cwd);

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

          const cardTitle = toSafeCardTitle(fm, slug);
          const svg = buildOgSvg({
            title: cardTitle,
            slug,
            published: fm.published,
            contentType: contentTypes.get(slug),
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

        // Also generate OG cards for standalone single-model report notes in src/content/tech-reports/
        const reportsDir = resolve(contentRoot, "tech-reports");
        try {
          const reportFiles = await fs.readdir(reportsDir, { withFileTypes: true });
          for (const file of reportFiles) {
            if (!file.isFile() || !/\.(md|mdx)$/.test(file.name)) continue;
            const slug = file.name.replace(/\.(md|mdx)$/, "");
            const raw = await fs.readFile(resolve(reportsDir, file.name), "utf-8");
            const fm = extractFrontmatter(raw);
            if (fm.draft === "true" || fm.seoThumbImage) continue;

            const cardTitle = toSafeCardTitle(fm, slug);
            const svg = buildOgSvg({
              title: cardTitle,
              slug: `reports/${slug}`,
              published: fm.published || fm.release,
              contentType: "reports",
            });
            try {
              await sharp(Buffer.from(svg))
                .png({ compressionLevel: 9 })
                .toFile(resolve(outDir, `reports-${slug}.png`));
              generated += 1;
            } catch (err) {
              console.warn(`[og-images] failed for reports/${slug}:`, err?.message || err);
            }
          }
        } catch {}

        console.log(`[og-images] Generated ${generated} OG card images in dist/og/`);
      },
    },
  };
}
