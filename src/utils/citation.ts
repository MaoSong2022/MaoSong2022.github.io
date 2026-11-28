import { SITE_AUTHOR, SITE_AUTHOR_CITATION, SITE_URL } from "../consts";

const DISPLAY_TIMEZONE = "Asia/Shanghai";

export interface CitationInput {
  title: string;
  authors: string[];
  published?: string | Date;
  url: string;
  slug?: string;
  doi?: string;
  siteHost?: string;
}

export interface CitationResult {
  citationText: string;
  bibtex: string;
  bibKey: string;
  year?: number;
  month?: string;
}

export interface CitationDateParts {
  year?: number;
  month?: string;
}

const stripHtml = (text: string): string => String(text || "").replace(/<[^>]*>/g, "");

export function flattenCitationTitle(rawTitle: string): string {
  return stripHtml(rawTitle)
    .replace(/\\n/g, " ")
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function formatCitationAuthor(name: string): string {
  const trimmed = String(name || "").trim();
  if (!trimmed) return "";
  if (trimmed.includes(",")) return trimmed;
  if (trimmed.toLowerCase() === SITE_AUTHOR.toLowerCase()) {
    return SITE_AUTHOR_CITATION;
  }
  return trimmed;
}

export function extractAuthorSurnameKey(name: string): string {
  const formatted = formatCitationAuthor(name);
  if (!formatted) return "article";
  const surname = formatted.includes(",")
    ? formatted.split(",")[0].trim()
    : formatted.split(/\s+/).slice(-1)[0].trim();
  const normalized = surname.toLowerCase().replace(/[^a-z0-9]+/g, "");
  return normalized || "article";
}

export function extractCitationDate(
  val: string | Date | undefined,
): CitationDateParts {
  if (!val) return {};

  if (typeof val === "string") {
    const trimmed = val.trim();
    if (!trimmed) return {};
    if (/^(19|20)\d{2}$/.test(trimmed)) {
      return { year: Number(trimmed) };
    }
    const yearMonthMatch = trimmed.match(/^((?:19|20)\d{2})[-/](0?[1-9]|1[0-2])$/);
    if (yearMonthMatch) {
      const year = Number(yearMonthMatch[1]);
      const monthIdx = Number(yearMonthMatch[2]) - 1;
      const month = new Date(Date.UTC(year, monthIdx, 15)).toLocaleString("en-US", {
        month: "short",
        timeZone: "UTC",
      });
      return { year, month };
    }
  }

  const date = val instanceof Date ? val : new Date(val);
  if (!Number.isNaN(date.getTime())) {
    const year = Number(
      date.toLocaleString("en-US", {
        year: "numeric",
        timeZone: DISPLAY_TIMEZONE,
      }),
    );
    const month = date.toLocaleString("en-US", {
      month: "short",
      timeZone: DISPLAY_TIMEZONE,
    });
    return { year, month };
  }

  const fallbackYear = String(val).match(/(19|20)\d{2}/);
  return fallbackYear ? { year: Number(fallbackYear[0]) } : {};
}

function resolveSiteHost(url: string, explicitHost?: string): string {
  if (explicitHost && explicitHost.trim()) return explicitHost.trim();
  try {
    return new URL(url).host;
  } catch {
    return new URL(SITE_URL).host;
  }
}

function sanitizeKeySegment(value: string | undefined): string {
  if (!value) return "";
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function buildBibKey(
  firstAuthor: string | undefined,
  year: number | undefined,
  slug: string | undefined,
  titleFlat: string,
): string {
  const keyAuthor = extractAuthorSurnameKey(firstAuthor || SITE_AUTHOR);
  const keySuffix =
    sanitizeKeySegment(slug) || sanitizeKeySegment(titleFlat) || "post";
  const yearPart = year != null ? String(year) : "";
  const separator =
    yearPart && /^[0-9]/.test(keySuffix)
      ? "_"
      : !yearPart
        ? "_"
        : "";
  return `${keyAuthor}${yearPart}${separator}${keySuffix}`;
}

export function buildArticleCitation(input: CitationInput): CitationResult {
  const titleFlat = flattenCitationTitle(input.title || "Untitled article");
  const rawAuthors = (input.authors || [])
    .map((a) => String(a || "").trim())
    .filter(Boolean);
  const formattedAuthors = (
    rawAuthors.length > 0 ? rawAuthors : [SITE_AUTHOR]
  ).map(formatCitationAuthor);

  const { year, month } = extractCitationDate(input.published);
  const siteHost = resolveSiteHost(input.url, input.siteHost);
  const doi = input.doi?.trim() || undefined;

  const hasCommaAuthor = formattedAuthors.some((a) => a.includes(","));
  const authorsPlain = formattedAuthors.join(hasCommaAuthor ? "; " : ", ");
  const datePlain =
    year && month ? ` (${month} ${year})` : year ? ` (${year})` : "";
  const authorAndDate = datePlain
    ? `${authorsPlain}.${datePlain}.`
    : `${authorsPlain}.`;
  const doiPlain = doi ? ` DOI: https://doi.org/${doi}` : "";
  const citationText = `${authorAndDate} "${titleFlat}". ${siteHost}. ${input.url}${doiPlain}`;

  const authorsBib = formattedAuthors.join(" and ");
  const bibKey = buildBibKey(
    formattedAuthors[0],
    year,
    input.slug,
    titleFlat,
  );

  const bibFields: Array<[string, string]> = [
    ["title", `{${titleFlat}}`],
    ["author", authorsBib],
    ...(year ? ([["year", String(year)]] as Array<[string, string]>) : []),
    ...(month ? ([["month", month]] as Array<[string, string]>) : []),
    ["howpublished", siteHost],
    ...(doi ? ([["doi", doi]] as Array<[string, string]>) : []),
    ["url", input.url],
  ];

  const maxFieldWidth = Math.max(...bibFields.map(([k]) => k.length));
  const bibLines = bibFields.map(
    ([key, value]) => `  ${key.padEnd(maxFieldWidth)} = {${value}}`,
  );
  const bibtex = `@misc{${bibKey},\n${bibLines.join(",\n")}\n}`;

  return {
    citationText,
    bibtex,
    bibKey,
    year,
    month,
  };
}
