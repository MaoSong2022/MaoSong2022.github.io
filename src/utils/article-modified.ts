import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readdir } from "node:fs/promises";
import { join, relative } from "node:path";

const CONTENT_ROOT = join(process.cwd(), "src", "content");
const MARKDOWN_EXTENSIONS = new Set([".md", ".mdx"]);
const SKIP_LASTMOD_RE = /\[skip-lastmod\]/i;

let cachedBundleModifiedPromise: Promise<Record<string, string>> | null = null;

function hasMarkdownExtension(filename: string): boolean {
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return false;
  return MARKDOWN_EXTENSIONS.has(filename.slice(dot));
}

function toPosixRepoPath(fullPath: string): string {
  return relative(process.cwd(), fullPath).replaceAll("\\", "/");
}

async function readMarkdownFilesRecursive(dirPath: string): Promise<string[]> {
  try {
    const entries = await readdir(dirPath, { withFileTypes: true });
    const files: string[] = [];

    for (const entry of entries) {
      const fullPath = join(dirPath, entry.name);
      if (entry.isDirectory()) {
        files.push(...(await readMarkdownFilesRecursive(fullPath)));
        continue;
      }
      if (entry.isFile() && hasMarkdownExtension(entry.name)) {
        files.push(fullPath);
      }
    }
    return files;
  } catch (error) {
    const code =
      error && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : "";
    if (code === "ENOENT") return [];
    throw error;
  }
}

async function collectBundleMarkdownFiles(): Promise<Map<string, string>> {
  const pathToSlug = new Map<string, string>();
  const entries = await readdir(CONTENT_ROOT, { withFileTypes: true });

  await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const slug = entry.name;
        const bundleRoot = join(CONTENT_ROOT, slug);
        const articleCandidates = [
          join(bundleRoot, "article.md"),
          join(bundleRoot, "article.mdx"),
        ];
        const chapterFiles = await readMarkdownFilesRecursive(
          join(bundleRoot, "chapters"),
        );
        for (const fullPath of [...articleCandidates, ...chapterFiles]) {
          if (existsSync(fullPath)) {
            pathToSlug.set(toPosixRepoPath(fullPath), slug);
          }
        }
      }),
  );

  return pathToSlug;
}

/**
 * Scan git history once with rename detection (`-M`), skipping pure `R100`
 * renames and mechanical maintenance commits tagged with `[skip-lastmod]`,
 * while continuing to follow paths backwards across renames.
 */
async function computeAllBundleLastModifiedIso(): Promise<Record<string, string>> {
  const pathToSlug = await collectBundleMarkdownFiles();
  const remainingSlugs = new Set(pathToSlug.values());
  const out: Record<string, string> = {};

  if (remainingSlugs.size === 0) return out;

  let rawLog = "";
  try {
    rawLog = execFileSync(
      "git",
      [
        "-c",
        "core.quotepath=false",
        "-c",
        "diff.renameLimit=5000",
        "log",
        "-M",
        "--name-status",
        "--format=__COMMIT__\x1f%cI\x1f%B\x1e",
        "--",
        "src/content/",
      ],
      {
        encoding: "utf-8",
        timeout: 30000,
        maxBuffer: 32 * 1024 * 1024,
        stdio: ["pipe", "pipe", "ignore"],
      },
    );
  } catch {
    return out;
  }

  const commits = rawLog.split("__COMMIT__\x1f");
  for (let i = 1; i < commits.length; i++) {
    if (remainingSlugs.size === 0) break;

    const chunk = commits[i];
    const metaEnd = chunk.indexOf("\x1e");
    if (metaEnd < 0) continue;

    const meta = chunk.slice(0, metaEnd);
    const sep = meta.indexOf("\x1f");
    if (sep < 0) continue;

    const commitDate = meta.slice(0, sep).trim();
    const commitMessage = meta.slice(sep + 1);
    const skipLastmod = SKIP_LASTMOD_RE.test(commitMessage);

    const statusLines = chunk
      .slice(metaEnd + 1)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    for (const line of statusLines) {
      const parts = line.split("\t");
      const status = parts[0] ?? "";

      if (status.startsWith("R") && parts.length >= 3) {
        const oldPath = parts[1];
        const newPath = parts[2];
        const slug = pathToSlug.get(newPath);
        if (!slug) continue;

        // Track historical path so earlier edits to the old path map to the current slug.
        pathToSlug.set(oldPath, slug);

        const isPureRename = status === "R100";
        if (!isPureRename && !skipLastmod && !out[slug] && commitDate) {
          out[slug] = commitDate;
          remainingSlugs.delete(slug);
        }
        continue;
      }

      if ((status === "A" || status === "M") && parts.length >= 2) {
        const filePath = parts[1];
        const slug = pathToSlug.get(filePath);
        if (slug && !skipLastmod && !out[slug] && commitDate) {
          out[slug] = commitDate;
          remainingSlugs.delete(slug);
        }
      }
    }
  }

  return out;
}

export async function getAllArticleBundleLastModifiedIso(): Promise<
  Record<string, string>
> {
  if (!cachedBundleModifiedPromise) {
    cachedBundleModifiedPromise = computeAllBundleLastModifiedIso();
  }
  return cachedBundleModifiedPromise;
}

export async function getArticleBundleLastModifiedIso(
  slug: string,
): Promise<string | undefined> {
  if (!slug) return undefined;
  const all = await getAllArticleBundleLastModifiedIso();
  return all[slug];
}

export async function getFileLastModifiedIso(
  relativePathFromContent: string,
): Promise<string | undefined> {
  const fullPath = join(CONTENT_ROOT, relativePathFromContent);
  if (!existsSync(fullPath)) return undefined;
  try {
    const output = execFileSync(
      "git",
      [
        "-c",
        "core.quotepath=false",
        "log",
        "-1",
        "--follow",
        "--diff-filter=AM",
        "-M",
        "--invert-grep",
        "--grep=\\[skip-lastmod\\]",
        "--format=%cI",
        "--",
        fullPath,
      ],
      {
        encoding: "utf-8",
        timeout: 10000,
        stdio: ["pipe", "pipe", "ignore"],
      },
    ).trim();
    return output || undefined;
  } catch {
    return undefined;
  }
}
