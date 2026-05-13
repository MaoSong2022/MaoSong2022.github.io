export type SiteNewsItem = {
  /** Shown as a short date line (e.g. ISO date or “May 2026”). */
  date: string;
  title: string;
  /** Optional one-line blurb under the title. */
  summary?: string;
  /** Link target when the headline should be clickable (site path or full URL). */
  href?: string;
};

/** Newest items first. Edit this list to update the home page news block. */
export const SITE_NEWS: SiteNewsItem[] = [
  {
    date: "2026-05-13",
    title: "Make this blog framework as a blog template",
    summary: "A blog template for research articles and blog posts.",
    href: "https://github.com/MaoSong2022/astro-blog-template",
  },
  {
    date: "2026-05-04",
    title: "Update the blog framework from Hugo to Astro",
    summary:
      "I update the blog framework from Hugo to Astro. Now the blog is more flexible and easier to maintain.",
    href: "/blog",
  },
  {
    date: "2026-01-26",
    title: "Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders",
    summary: "We invalidate the hypothesis that 'adding more vision encoders will always improve the performance of multimodal large language models'.\n Accetpted by ICLR2026.",
    href: "https://maosong.website/Encoder-Redundancy/",
  },
];
