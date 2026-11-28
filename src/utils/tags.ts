/**
 * Shared tag normalization and case-insensitive deduplication helpers.
 *
 * Canonical tag conventions across articles and single-report notes:
 * - Title Case with canonical acronyms/proper nouns preserved (e.g. "Long Context",
 *   "Position Encoding", "Agentic RL", "Pre-training", "Post-training", "MoE", "Qwen").
 * - YAML frontmatter uses a double-quoted block list (`- "Tag"`).
 * - Model-series report articles use canonical model series names (e.g. "Qwen", "DeepSeek",
 *   "Gemini", "Gemma", "GLM", "GPT", "Kimi", "LLaMA", "Mistral", "Nemotron", "OLMo").
 * - High-level subjects ("LLM", "MLLM", "RL", "Infra", "Math") belong in `categories`,
 *   and content types belong in `ARTICLE_EDITORIAL` — avoid meta or overly narrow tags.
 */

const MULTI_SPACE_RE = /\s+/g;

export function normalizeTagValue(raw: unknown): string {
  if (typeof raw === "string") {
    return raw.trim().replace(MULTI_SPACE_RE, " ");
  }
  if (raw != null && typeof raw !== "object") {
    return String(raw).trim().replace(MULTI_SPACE_RE, " ");
  }
  return "";
}

/**
 * Normalize raw frontmatter `tags` (string or string[]) into a clean,
 * case-insensitively deduplicated string array preserving first-seen order.
 */
export function normalizeTagList(raw: unknown): string[] {
  if (raw == null) return [];
  const items = Array.isArray(raw) ? raw : [raw];
  const seen = new Set<string>();
  const out: string[] = [];

  for (const item of items) {
    const tag = normalizeTagValue(item);
    if (!tag) continue;
    const key = tag.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(tag);
  }

  return out;
}

/**
 * Collect unique tags across multiple tag lists, deduplicating case-insensitively
 * and sorting alphabetically (case-insensitive, English locale).
 */
export function collectUniqueTags(
  tagLists: Iterable<readonly string[]>,
): string[] {
  const byKey = new Map<string, string>();
  for (const list of tagLists) {
    for (const raw of list) {
      const tag = normalizeTagValue(raw);
      if (!tag) continue;
      const key = tag.toLowerCase();
      if (!byKey.has(key)) {
        byKey.set(key, tag);
      }
    }
  }
  return [...byKey.values()].sort((a, b) =>
    a.localeCompare(b, "en", { sensitivity: "base" }),
  );
}
