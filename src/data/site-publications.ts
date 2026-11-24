export type PublicationItem = {
  title: string;
  venue: string;
  summary: string;
  links: Array<{ label: string; href: string }>;
};

/** Selected research publications shown on the homepage. */
export const SITE_PUBLICATIONS: PublicationItem[] = [
  {
    title:
      "Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders",
    venue: "ICLR 2026",
    summary:
      "We invalidate the hypothesis that adding more vision encoders always improves MLLM performance and systematically analyze multi-encoder redundancy.",
    links: [
      {
        label: "Project Page",
        href: "https://maosong.website/Encoder-Redundancy/",
      },
    ],
  },
  {
    title:
      "DocGenome: An Open Large-scale Scientific Document Benchmark for Training and Testing Multi-modal Large Language Models",
    venue: "arXiv 2024",
    summary:
      "An open large-scale scientific document benchmark built on 500K arXiv papers across 153 disciplines, providing structured layout, logical relations, and LaTeX annotations for training and evaluating MLLMs.",
    links: [
      {
        label: "Project Page",
        href: "https://alpha-innovator.github.io/DocGenome_page/",
      },
      {
        label: "arXiv",
        href: "https://arxiv.org/abs/2406.11633",
      },
    ],
  },
];
