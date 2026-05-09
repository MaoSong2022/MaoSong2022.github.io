export type SiteArticleEntry = {
  slug: string;
  url: string;
  title: string;
  published?: string;
  description?: string;
  /** Trimmed strings from frontmatter `tags`. */
  tags: string[];
  /** From frontmatter `category` or first entry of `categories`. */
  category?: string;
};

const stripHtml = (text: string) =>
  String(text || "").replace(/<[^>]*>/g, "");

const flatTitle = (raw: unknown): string => {
  const s =
    typeof raw === "string"
      ? raw
      : raw != null
        ? String(raw)
        : "Untitled";
  return stripHtml(s).replace(/\\n/g, " ").replace(/\n/g, " ").trim();
};

function normalizeTags(raw: unknown): string[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    const out: string[] = [];
    for (const x of raw) {
      if (typeof x === "string" && x.trim()) out.push(x.trim());
      else if (x != null && typeof x !== "object")
        out.push(String(x).trim());
    }
    return out.filter(Boolean);
  }
  if (typeof raw === "string" && raw.trim()) return [raw.trim()];
  return [];
}

function normalizeCategory(raw: unknown): string | undefined {
  if (typeof raw === "string" && raw.trim()) return raw.trim();
  if (Array.isArray(raw)) {
    const first = raw[0];
    if (typeof first === "string" && first.trim()) return first.trim();
  }
  return undefined;
}

/** Normalize frontmatter from eager `import.meta.glob` over each `content/<slug>/article.{md,mdx}` module. */
export function parseArticleGlob(
  articleGlob: Record<string, { frontmatter?: Record<string, unknown> }>,
): SiteArticleEntry[] {
  const articles: SiteArticleEntry[] = Object.entries(articleGlob).map(
    ([path, mod]) => {
      const match = path.match(/content\/([^/]+)\/article\.(?:md|mdx)$/);
      const slug = match?.[1] ?? "unknown";
      const fm = mod?.frontmatter ?? {};
      const title = flatTitle(fm.title) || slug;
      const published =
        typeof fm.published === "string" ? fm.published : undefined;
      const description =
        typeof fm.description === "string" ? fm.description : undefined;
      const tags = normalizeTags(fm.tags);
      const category = normalizeCategory(fm.category ?? fm.categories);
      return {
        slug,
        url: `/p/${slug}/`,
        title,
        published,
        description,
        tags,
        category,
      };
    },
  );

  articles.sort((a, b) => {
    const ta = Date.parse(a.published ?? "") || 0;
    const tb = Date.parse(b.published ?? "") || 0;
    if (tb !== ta) return tb - ta;
    return a.slug.localeCompare(b.slug);
  });

  return articles;
}
