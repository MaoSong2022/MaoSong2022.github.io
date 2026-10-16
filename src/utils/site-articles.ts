import { formatDisplayDate } from "./date-format";
import { ARTICLE_EDITORIAL, type Editorial, type ContentType, type Completion } from "../data/article-editorial";

export type SiteArticleEntry = Editorial & {
  classified: boolean;
  slug: string;
  url: string;
  title: string;
  published?: string;
  modified?: string;
  displayDate?: string;
  description?: string;
  /** Trimmed strings from frontmatter `tags`. */
  tags: string[];
  /** From frontmatter `categories`. */
  categories?: string;
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

function normalizeFirstCategory(raw: unknown): string | undefined {
  if (typeof raw === "string" && raw.trim()) return raw.trim();
  if (Array.isArray(raw)) {
    const first = raw[0];
    if (typeof first === "string" && first.trim()) return first.trim();
  }
  return undefined;
}

/** Normalize frontmatter from eager `import.meta.glob` over each `content/<slug>/article.{md,mdx}` module. */
export async function parseArticleGlob(
  articleGlob: Record<string, { frontmatter?: Record<string, unknown> }>,
  modifiedBySlug: Record<string, string> = {},
): Promise<SiteArticleEntry[]> {
  const articles: SiteArticleEntry[] = Object.entries(articleGlob).filter(([path]) => !/content\/demo\/article\.(md|mdx)$/.test(path)).map(
    ([path, mod]): SiteArticleEntry => {
      const match = path.match(/content\/([^/]+)\/article\.(?:md|mdx)$/);
      const slug = match?.[1] ?? "unknown";
      const fm = mod?.frontmatter ?? {};
      const title = flatTitle(fm.title) || slug;
      const published =
        typeof fm.published === "string" ? fm.published : undefined;
      const frontmatterModified =
        typeof fm.modified === "string"
          ? fm.modified
          : typeof fm.updatedDate === "string"
            ? fm.updatedDate
            : typeof fm.lastmod === "string"
              ? fm.lastmod
              : undefined;
      const modified = modifiedBySlug[slug] ?? frontmatterModified ?? published;
      const description =
        typeof fm.description === "string" ? fm.description : undefined;
      const tags = normalizeTags(fm.tags);
      const categories = normalizeFirstCategory(fm.categories);
      return {
        ...(ARTICLE_EDITORIAL[slug] ?? { contentType: "learning", completion: "outline" }),
        classified: Object.hasOwn(ARTICLE_EDITORIAL, slug),
        slug,
        url: `/blog/${slug}/`,
        title,
        published,
        modified,
        displayDate: formatDisplayDate(modified ?? published) ?? modified ?? published,
        description,
        tags,
        categories,
      };
    },
  );

  articles.sort((a, b) => {
    const ta = Date.parse(a.modified ?? a.published ?? "") || 0;
    const tb = Date.parse(b.modified ?? b.published ?? "") || 0;
    if (tb !== ta) return tb - ta;
    return a.slug.localeCompare(b.slug);
  });

  return articles;
}

/** Stable sorting preserves the existing newest-first order within each status. */
export function sortByCompletion<T extends { completion: Completion }>(articles: T[]): T[] {
  const rank = { complete: 0, growing: 1, outline: 2 };
  return [...articles].sort((a, b) => rank[a.completion] - rank[b.completion]);
}

export function selectFeatured(articles: SiteArticleEntry[], type: ContentType): SiteArticleEntry[] {
  return articles.filter(a => a.contentType === type && a.featuredOrder !== undefined
    && a.completion !== "outline" && (type !== "learning" || a.completion === "complete"))
    .sort((a, b) => a.featuredOrder! - b.featuredOrder!);
}
