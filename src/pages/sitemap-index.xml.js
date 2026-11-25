export async function GET(context) {
  const site = new URL(context.site);
  const sitemapUrl = new URL("/sitemap.xml", site).toString();

  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <sitemap>\n    <loc>${sitemapUrl}</loc>\n  </sitemap>\n</sitemapindex>\n`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
}
