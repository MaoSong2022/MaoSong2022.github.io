export type TechReportKind = "note" | "brief";

export type TechReportEntry = {
  /** URL slug under /blog/reports/<slug>/ */
  slug: string;
  /** Model display name */
  name: string;
  /** Organization or team */
  org: string;
  /** Release month (YYYY-MM) */
  release: string;
  /** Quarterly summary article slug under /blog/<digest>/ */
  digest: string;
  /** Human-readable quarter label, e.g. "2026 Q4" */
  quarter: string;
  /** Base model or architecture scale */
  baseOrArch: string;
  /** Reading / note date (YYYY-MM-DD) */
  published?: string;
  /** Non-exclusive technical tags for cross-quarter filtering */
  tags: string[];
  /** Concise one-line takeaway */
  summary: string;
  /** Full reading note vs short brief */
  kind: TechReportKind;
  /** Standalone page URL */
  url: string;
};

type RawReportModule = {
  frontmatter?: Record<string, unknown>;
};

const reportGlob = import.meta.glob<RawReportModule>(
  "../content/tech-reports/*.{md,mdx}",
  { eager: true },
);

function normalizeStringArray(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw
      .map((v) => (typeof v === "string" ? v.trim() : String(v ?? "").trim()))
      .filter(Boolean);
  }
  if (typeof raw === "string" && raw.trim()) return [raw.trim()];
  return [];
}

function parseTechReports(): TechReportEntry[] {
  const entries: TechReportEntry[] = [];

  for (const [path, mod] of Object.entries(reportGlob)) {
    const match = path.match(/tech-reports\/([^/]+)\.(?:md|mdx)$/);
    if (!match) continue;
    const slug = match[1];
    const fm = mod?.frontmatter ?? {};
    if (fm.draft === true || fm.draft === "true") continue;

    const kind: TechReportKind = fm.kind === "brief" ? "brief" : "note";
    const name = typeof fm.title === "string" && fm.title.trim() ? fm.title.trim() : slug;
    const org = typeof fm.org === "string" ? fm.org.trim() : "";
    const release = typeof fm.release === "string" ? fm.release.trim() : "";
    const digest = typeof fm.digest === "string" ? fm.digest.trim() : "";
    const quarter = typeof fm.quarter === "string" ? fm.quarter.trim() : "";
    const baseOrArch = typeof fm.baseOrArch === "string" ? fm.baseOrArch.trim() : "";
    const published = typeof fm.published === "string" ? fm.published.trim() : undefined;
    const summary =
      typeof fm.summary === "string" && fm.summary.trim()
        ? fm.summary.trim()
        : typeof fm.description === "string"
          ? fm.description.trim()
          : "";
    const tags = normalizeStringArray(fm.tags);

    entries.push({
      slug,
      name,
      org,
      release,
      digest,
      quarter,
      baseOrArch,
      published,
      tags,
      summary,
      kind,
      url: reportHref(slug),
    });
  }

  // Full notes first, then newest reading/release date first
  entries.sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === "note" ? -1 : 1;
    const pa = Date.parse(a.published ?? "") || 0;
    const pb = Date.parse(b.published ?? "") || 0;
    if (pb !== pa) return pb - pa;
    if (b.release !== a.release) return b.release.localeCompare(a.release);
    return a.name.localeCompare(b.name);
  });

  return entries;
}

export function reportHref(slug: string): string {
  return `/blog/reports/${slug}/`;
}

export function getAllTechReports(): TechReportEntry[] {
  return parseTechReports();
}

export function getTechReportsByDigest(digestSlug: string): TechReportEntry[] {
  return parseTechReports().filter((r) => r.digest === digestSlug);
}

export function getAllReportTags(): string[] {
  return [...new Set(parseTechReports().flatMap((r) => r.tags))].sort((a, b) =>
    a.localeCompare(b),
  );
}

export function getAllReportQuarters(): Array<{ digest: string; quarter: string }> {
  const seen = new Map<string, string>();
  for (const r of parseTechReports()) {
    if (r.digest && r.quarter && !seen.has(r.digest)) {
      seen.set(r.digest, r.quarter);
    }
  }
  return Array.from(seen.entries())
    .map(([digest, quarter]) => ({ digest, quarter }))
    .sort((a, b) => b.quarter.localeCompare(a.quarter));
}

/**
 * Extra search keywords injected into /browse for a quarterly review article
 * so searching any individual model name, org, or tag surfaces its quarter.
 */
export function getDigestSearchText(digestSlug: string): string {
  const rows = getTechReportsByDigest(digestSlug);
  if (!rows.length) return "";
  return rows
    .flatMap((r) => [r.name, r.org, r.baseOrArch, r.release, ...r.tags, r.summary])
    .join(" ");
}
