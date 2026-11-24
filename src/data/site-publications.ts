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
];
