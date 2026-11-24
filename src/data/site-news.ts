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
    date: "2026-09-01",
    title: "Joined SenseTime as a Researcher",
    summary: "I joined SenseTime as a Researcher, focusing on LLM post-training.",
  },
  {
    date: "2026-05-13",
    title: "Released this blog framework as an Astro template",
    summary: "An open-source Astro template for research articles and technical blogs.",
    href: "https://github.com/MaoSong2022/astro-blog-template",
  },
  {
    date: "2026-05-04",
    title: "Migrated the blog from Hugo to Astro",
    summary:
      "Migrated the blog framework from Hugo to Astro for better flexibility and maintainability.",
    href: "/blog",
  },
  {
    date: "2026-01-26",
    title: "Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders",
    summary: "We invalidate the hypothesis that 'adding more vision encoders will always improve the performance of multimodal large language models'.\n Accepted to ICLR 2026.",
    href: "https://maosong.website/Encoder-Redundancy/",
  },
];
