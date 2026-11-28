import { promises as fs } from "node:fs";
import { resolve } from "node:path";

export interface ReadingTimeInfo {
  minutes: number;
  words: number;
  label: string;
  title: string;
}

const CONTENT_ROOT = resolve(process.cwd(), "src", "content");
const CJK_CHARS_PER_MIN = 320;
const LATIN_WORDS_PER_MIN = 200;
const CODE_WORDS_PER_MIN = 400;

const bundleCache = new Map<string, Promise<ReadingTimeInfo | undefined>>();
const fileCache = new Map<string, Promise<ReadingTimeInfo | undefined>>();

async function readTextSafe(filePath: string): Promise<string> {
  try {
    return await fs.readFile(filePath, "utf-8");
  } catch {
    return "";
  }
}

function stripFrontmatter(raw: string): string {
  return raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
}

async function resolveImportedChapters(
  body: string,
  bundleDir: string,
): Promise<string> {
  // Drop JSX comments first so commented-out chapter tags/imports are ignored.
  let out = body.replace(/\{\/\*[\s\S]*?\*\/\}/g, "");
  const importRe =
    /^import\s+(\w+)\s+from\s+["']\.\/chapters\/([^"']+)["'];?\s*$/gm;
  const imports = new Map<string, string>();
  let m: RegExpExecArray | null;
  while ((m = importRe.exec(out)) !== null) {
    imports.set(m[1], m[2]);
  }
  out = out.replace(importRe, "");

  for (const [name, relPath] of imports) {
    const tagRe = new RegExp(`<${name}\\s*/?>`, "g");
    if (!tagRe.test(out)) continue;
    const chapterRaw = await readTextSafe(resolve(bundleDir, "chapters", relPath));
    if (!chapterRaw) continue;
    const chapterBody = stripFrontmatter(chapterRaw);
    out = out.replace(new RegExp(`<${name}\\s*/>`, "g"), chapterBody);
  }

  return out;
}

export function estimateReadingTime(markdownBody: string): ReadingTimeInfo {
  let text = stripFrontmatter(markdownBody);

  // Strip comments and ESM import/export lines.
  text = text.replace(/\{\/\*[\s\S]*?\*\/\}/g, " ");
  text = text.replace(/<!--[\s\S]*?-->/g, " ");
  text = text.replace(/^(?:import|export)\s+.*$/gm, " ");

  // Extract fenced code blocks separately so code is weighted more lightly.
  let codeWords = 0;
  text = text.replace(/```[\s\S]*?```/g, (block) => {
    const inner = block.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
    const tokens = inner.match(/[A-Za-z0-9_]+/g);
    codeWords += tokens ? tokens.length : 0;
    return " ";
  });

  // Remove images, keep link labels, drop raw URLs.
  text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
  text = text.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
  text = text.replace(/https?:\/\/\S+/g, " ");

  // Strip HTML / JSX tags and LaTeX command names (keeping readable text/symbols).
  text = text.replace(/<[^>]+>/g, " ");
  text = text.replace(/\\[a-zA-Z]+/g, " ");

  const cjkMatches = text.match(/[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/g);
  const cjkCount = cjkMatches ? cjkMatches.length : 0;

  const nonCjkText = text.replace(
    /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/g,
    " ",
  );
  const latinMatches = nonCjkText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g);
  const latinCount = latinMatches ? latinMatches.length : 0;

  const totalMinutes =
    cjkCount / CJK_CHARS_PER_MIN +
    latinCount / LATIN_WORDS_PER_MIN +
    codeWords / CODE_WORDS_PER_MIN;

  const minutes = Math.max(1, Math.round(totalMinutes));
  const words = cjkCount + latinCount;
  const formattedWords =
    words >= 1000
      ? `${(words / 1000).toFixed(1).replace(/\.0$/, "")}k`
      : String(words);

  return {
    minutes,
    words,
    label: `${minutes} min`,
    title: `约 ${formattedWords} 字 · 预计阅读 ${minutes} 分钟`,
  };
}

export function getArticleBundleReadingTime(
  slug: string,
): Promise<ReadingTimeInfo | undefined> {
  if (!slug) return Promise.resolve(undefined);
  const existing = bundleCache.get(slug);
  if (existing) return existing;

  const task = (async () => {
    const bundleDir = resolve(CONTENT_ROOT, slug);
    let raw = "";
    for (const filename of ["article.mdx", "article.md"]) {
      raw = await readTextSafe(resolve(bundleDir, filename));
      if (raw) break;
    }
    if (!raw) return undefined;
    const expanded = await resolveImportedChapters(
      stripFrontmatter(raw),
      bundleDir,
    );
    return estimateReadingTime(expanded);
  })();

  bundleCache.set(slug, task);
  return task;
}

export function getContentFileReadingTime(
  relativePathFromContent: string,
): Promise<ReadingTimeInfo | undefined> {
  if (!relativePathFromContent) return Promise.resolve(undefined);
  const existing = fileCache.get(relativePathFromContent);
  if (existing) return existing;

  const task = (async () => {
    const raw = await readTextSafe(resolve(CONTENT_ROOT, relativePathFromContent));
    if (!raw) return undefined;
    return estimateReadingTime(raw);
  })();

  fileCache.set(relativePathFromContent, task);
  return task;
}
