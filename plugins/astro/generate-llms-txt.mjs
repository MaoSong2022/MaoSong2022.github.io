/**
 * Astro integration: generate /llms.txt and per-article /blog/<slug>/llms.txt
 * at build time following the llms.txt specification (https://llmstxt.org/).
 *
 * Scans src/content/<slug>/article.{md,mdx}, excludes demo and draft articles,
 * resolves chapter imports, strips JSX/components, and outputs clean Markdown
 * for AI crawlers and LLM assistants.
 */
import { promises as fs } from "node:fs";
import { resolve } from "node:path";

async function readFile(filePath) {
  try {
    return await fs.readFile(filePath, "utf-8");
  } catch {
    return "";
  }
}

function stripQuotes(s) {
  return String(s || "")
    .replace(/^["']|["']$/g, "")
    .trim();
}

function extractFrontmatter(content) {
  const m = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { frontmatter: {}, body: content };

  const fm = {};
  const lines = m[1].split(/\r?\n/);
  let key = null;
  let val = "";
  let multi = false;

  for (const line of lines) {
    if (line.match(/^[a-zA-Z_]\w*\s*:/) && !multi) {
      if (key) fm[key] = val.trim();
      const [k, ...rest] = line.split(":");
      key = k.trim();
      val = rest.join(":").trim();
      if (val === ">" || val === "|") {
        multi = true;
        val = "";
      }
    } else if (key) {
      if (multi && line.match(/^[a-zA-Z_]\w*\s*:/)) {
        fm[key] = val.trim();
        multi = false;
        const [k, ...rest] = line.split(":");
        key = k.trim();
        val = rest.join(":").trim();
      } else {
        val += (multi ? " " : "\n") + line;
      }
    }
  }
  if (key) fm[key] = val.trim();

  return { frontmatter: fm, body: content.slice(m[0].length) };
}

function parseAuthors(raw) {
  if (!raw) return [];
  const names = [];
  const re = /name:\s*["']?([^"'\n]+)["']?/g;
  let hit;
  while ((hit = re.exec(raw)) !== null) names.push(hit[1].trim());
  if (names.length) return names;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function mdxToMarkdown(content) {
  let out = content;
  // Remove JSX comments {/* ... */}
  out = out.replace(/\{\/\*[\s\S]*?\*\/\}/g, "");
  // Remove import statements
  out = out.replace(/^import .+?;?\s*$/gm, "");
  // Replace Sidenote with blockquote
  out = out.replace(/<Sidenote>([\s\S]*?)<\/Sidenote>/g, (_, inner) => {
    const aside = inner.match(
      /<Fragment slot="aside">([\s\S]*?)<\/Fragment>/,
    );
    const main = inner
      .replace(/<Fragment slot="aside">[\s\S]*?<\/Fragment>/, "")
      .trim();
    return aside ? `${main}\n\n> ${aside[1].trim()}` : main;
  });
  out = out.replace(
    /<Note[^>]*>([\s\S]*?)<\/Note>/g,
    (_, inner) => `\n> ${inner.trim()}\n`,
  );
  out = out.replace(/<(Wide|FullWidth)>([\s\S]*?)<\/\1>/g, "$2");
  out = out.replace(
    /<HtmlEmbed\s+[^>]*src="([^"]*)"[^>]*\/>/g,
    "*[Interactive visualization: $1]*",
  );
  out = out.replace(/<Fragment[^>]*>([\s\S]*?)<\/Fragment>/g, "$1");
  out = out.replace(/<[A-Z][a-zA-Z0-9]*\s*\/>/g, "");
  out = out.replace(
    /<[A-Z][a-zA-Z0-9]*[^>]*>([\s\S]*?)<\/[A-Z][a-zA-Z0-9]*>/g,
    "$1",
  );
  out = out.replace(/className="[^"]*"/g, "");
  out = out.replace(/<br\s*\/?>/g, "\n");
  out = out.replace(/<details>\s*([\s\S]*?)<\/details>/g, "$1");
  out = out.replace(/<summary>([\s\S]*?)<\/summary>/g, "**$1**\n");
  out = out.replace(/<a\s+href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g, "[$2]($1)");
  out = out.replace(/<[^>]+>/g, "");
  out = out.replace(/ {2,}/g, " ");
  out = out.replace(/\n{3,}/g, "\n\n");
  return out.trim();
}

async function resolveChapters(content, bundleDir) {
  let out = content;
  // Remove JSX comments first so commented-out imports are ignored
  out = out.replace(/\{\/\*[\s\S]*?\*\/\}/g, "");
  const importRe =
    /^import\s+(\w+)\s+from\s+["']\.\/chapters\/([^"']+)["'];?\s*$/gm;
  const imports = new Map();
  let m;
  while ((m = importRe.exec(out)) !== null) {
    imports.set(m[1], m[2]);
  }
  out = out.replace(importRe, "");

  for (const [name, relPath] of imports) {
    const re = new RegExp(`<${name}\\s*/>`, "g");
    const raw = await readFile(resolve(bundleDir, "chapters", relPath));
    const { body } = extractFrontmatter(raw);
    out = out.replace(re, mdxToMarkdown(body));
  }
  return out;
}

function cleanTitle(raw, fallback) {
  const title = stripQuotes(
    String(raw || fallback || "Untitled")
      .replace(/<[^>]*>/g, "")
      .replace(/\\n/g, " ")
      .replace(/\n/g, " ")
      .replace(/ {2,}/g, " "),
  );
  return title || fallback || "Untitled";
}

function buildArticleHeader(fm, slug, siteUrl) {
  const title = cleanTitle(fm.title, slug);
  const parts = [`# ${title}\n`];

  const desc = stripQuotes(fm.description || fm.subtitle || "");
  if (desc) parts.push(`> ${desc}\n`);

  const meta = [`- **URL**: ${siteUrl}/blog/${slug}/`];
  const authors = parseAuthors(fm.authors);
  if (authors.length) meta.push(`- **Authors**: ${authors.join(", ")}`);
  if (fm.published) meta.push(`- **Published**: ${stripQuotes(fm.published)}`);
  if (fm.modified) meta.push(`- **Modified**: ${stripQuotes(fm.modified)}`);
  if (fm.doi) meta.push(`- **DOI**: https://doi.org/${fm.doi}`);
  parts.push(meta.join("\n") + "\n");

  parts.push("---\n");
  return parts.join("\n");
}

async function readSiteMeta(cwd) {
  const constsRaw = await readFile(resolve(cwd, "src/consts.ts"));
  const getConst = (name, fallback) => {
    const m = constsRaw.match(
      new RegExp(`export\\s+const\\s+${name}\\s*=\\s*(["'\`])([\\s\\S]*?)\\1`),
    );
    return m ? m[2].trim() : fallback;
  };
  return {
    siteUrl: getConst("SITE_URL", "https://maosong.website").replace(/\/+$/, ""),
    siteTitle: getConst("SITE_TITLE", "Mao Song"),
    siteAuthor: getConst("SITE_AUTHOR", "Mao Song"),
    siteDescription: getConst(
      "SITE_DESCRIPTION",
      "Technical blog covering machine learning, LLMs, deep learning research, and AI.",
    ),
  };
}

async function readEditorialMap(cwd) {
  const raw = await readFile(resolve(cwd, "src/data/article-editorial.ts"));
  const map = new Map();
  // Parse simple entries from ARTICLE_EDITORIAL
  const re =
    /^\s*(?:["']([^"']+)["']|([a-zA-Z0-9_-]+))\s*:\s*(.+?),\s*$/gm;
  let m;
  while ((m = re.exec(raw)) !== null) {
    const slug = m[1] || m[2];
    const rhs = m[3];
    let contentType = "learning";
    let completion = "growing";
    if (rhs.includes("report(") || rhs.includes('"reports"')) contentType = "reports";
    else if (rhs.includes("topic(") || rhs.includes('"topics"')) contentType = "topics";
    else if (rhs.includes('"analysis"')) contentType = "analysis";
    else if (rhs.includes("learning(") || rhs.includes('"learning"')) contentType = "learning";

    if (rhs.includes('"complete"')) completion = "complete";
    else if (rhs.includes('"outline"')) completion = "outline";
    else if (rhs.includes('"growing"')) completion = "growing";

    map.set(slug, { contentType, completion });
  }
  return map;
}

export async function buildLlmsTxtFiles({ cwd = process.cwd(), outDirPath }) {
  const contentRoot = resolve(cwd, "src/content");
  const { siteUrl, siteTitle, siteAuthor, siteDescription } = await readSiteMeta(cwd);
  const editorialMap = await readEditorialMap(cwd);

  const entries = await fs.readdir(contentRoot, { withFileTypes: true });
  const articles = [];

  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name === "demo") continue;
    const slug = entry.name;
    const bundleDir = resolve(contentRoot, slug);
    let raw = "";
    for (const ext of ["article.mdx", "article.md"]) {
      raw = await readFile(resolve(bundleDir, ext));
      if (raw) break;
    }
    if (!raw) continue;

    const { frontmatter, body } = extractFrontmatter(raw);
    const isDraft = stripQuotes(frontmatter.draft).toLowerCase() === "true";
    if (isDraft) continue;

    const editorial = editorialMap.get(slug) ?? {
      contentType: "learning",
      completion: "outline",
    };

    const withChapters = await resolveChapters(body, bundleDir);
    const markdown = mdxToMarkdown(withChapters);
    const header = buildArticleHeader(frontmatter, slug, siteUrl);
    const articleOutput = header + "\n" + markdown + "\n";

    const articleOutDir = resolve(outDirPath, "blog", slug);
    await fs.mkdir(articleOutDir, { recursive: true });
    await fs.writeFile(resolve(articleOutDir, "llms.txt"), articleOutput, "utf-8");

    articles.push({
      slug,
      title: cleanTitle(frontmatter.title, slug),
      description: stripQuotes(frontmatter.description || frontmatter.subtitle || ""),
      contentType: editorial.contentType,
      completion: editorial.completion,
    });
  }

  const sectionOrder = [
    ["analysis", "Deep Dives"],
    ["learning", "Learning Notes & Tutorials"],
    ["topics", "Topic Overviews"],
    ["reports", "Technical Reports"],
  ];

  const completionRank = { complete: 0, growing: 1, outline: 2 };
  const displaySiteName =
    siteTitle && siteTitle.toLowerCase() !== "homepage" ? siteTitle : siteAuthor;

  const indexLines = [
    `# ${displaySiteName}`,
    "",
    `> ${siteDescription}`,
    "",
    `- Author: ${siteAuthor}`,
    `- Website: ${siteUrl}/`,
    `- All Articles: ${siteUrl}/blog/`,
    `- RSS Feed: ${siteUrl}/rss.xml`,
    "",
  ];

  for (const [type, label] of sectionOrder) {
    const group = articles
      .filter((a) => a.contentType === type && a.completion !== "outline")
      .sort(
        (a, b) =>
          (completionRank[a.completion] ?? 9) - (completionRank[b.completion] ?? 9) ||
          a.title.localeCompare(b.title),
      );
    if (group.length === 0) continue;
    indexLines.push(`## ${label}`, "");
    for (const a of group) {
      const desc = a.description ? `: ${a.description}` : "";
      indexLines.push(
        `- [${a.title}](${siteUrl}/blog/${a.slug}/) ([Markdown](${siteUrl}/blog/${a.slug}/llms.txt))${desc}`,
      );
    }
    indexLines.push("");
  }

  const indexContent = indexLines.join("\n").trim() + "\n";
  await fs.writeFile(resolve(outDirPath, "llms.txt"), indexContent, "utf-8");

  return {
    articleCount: articles.length,
    indexBytes: Buffer.byteLength(indexContent),
  };
}

export default function generateLlmsTxt() {
  return {
    name: "generate-llms-txt",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        const { articleCount, indexBytes } = await buildLlmsTxtFiles({
          cwd: process.cwd(),
          outDirPath: dir.pathname,
        });
        const kb = (indexBytes / 1024).toFixed(1);
        console.log(
          `[llms.txt] Generated dist/llms.txt (${kb} KB) and ${articleCount} per-article llms.txt files`,
        );
      },
    },
  };
}
