/** Editorial purpose and reading readiness are intentionally separate. */
export const CONTENT_TYPES = {
  learning: { label: "Learning Notes & Tutorials", shortLabel: "Learning Notes", description: "Build the mathematical, algorithmic, and systems foundations for understanding LLMs." },
  reports: { label: "Technical Reports", shortLabel: "Technical Reports", description: "Reading notes on model architectures, training recipes, and experimental evidence." },
  topics: { label: "Topic Overviews", shortLabel: "Topic Overviews", description: "Follow how problems, methods, and trade-offs evolve within a research area." },
  analysis: { label: "Deep Dives", shortLabel: "Deep Dives", description: "Explore specific questions through derivations, comparisons, and independent analysis." },
} as const;

export const COMPLETION = {
  complete: { label: "Ready to Read", description: "A coherent treatment of the current scope, ready for end-to-end reading." },
  growing: { label: "In Progress", description: "Substantial content is available; sections, derivations, or validation are still being developed." },
  outline: { label: "Outline", description: "An outline, source collection, or early notes." },
} as const;

export type ContentType = keyof typeof CONTENT_TYPES;
export type Completion = keyof typeof COMPLETION;
export type Editorial = {
  contentType: ContentType;
  completion: Completion;
  /** Explicit order within a homepage section; never inferred from recency. */
  featuredOrder?: number;
  summary?: string;
};

const learning = (completion: Completion = "growing"): Editorial => ({ contentType: "learning", completion });
const report = (): Editorial => ({ contentType: "reports", completion: "growing" });
const topic = (completion: Completion = "growing"): Editorial => ({ contentType: "topics", completion });

export const ARTICLE_EDITORIAL: Record<string, Editorial> = {
  math: learning(),
  tokenizer: { ...learning("complete"), featuredOrder: 3, summary: "从文本到 token：理解分词方法，以及词表与编码的基本取舍。" },
  transformer: learning(),
  RL4LLM: { ...learning("complete"), featuredOrder: 1, summary: "从强化学习基础到大模型训练，串起价值估计、策略优化与关键推导。" },
  "llm-resource-computation": { ...learning("complete"), featuredOrder: 2, summary: "理解大模型参数量、计算量与显存开销，建立训练资源估算的基础。" },
  parallelism: learning(),
  "GPU-TPU": learning(),
  "performance-and-scalability": learning(),
  deep_learning: learning("outline"),
  "deepseek-series": report(),
  "Qwen-series": report(),
  kimi: report(),
  nemotron: report(),
  composer: report(),
  GLM: report(),
  minimax: report(),
  olmo: report(),
  "qwen-vl-series": report(),
  internvl: report(),
  "keye-vl": report(),
  mimo: report(),
  "LLaMA-series": report(),
  Gemma: report(),
  gemini: report(),
  gpt: report(),
  minicpm: report(),
  tech_report: report(),
  "deepseek-r1": { contentType: "reports", completion: "outline" },
  attention: topic(),
  "long-context": topic(),
  moe: topic(),
  OPD: topic(),
  "policy-optimization": topic(),
  inference: topic(),
  MLLM: topic(),
  "visual-foundation-model": topic(),
  optimizer: topic(),
  "position-encoding": topic(),
  FlashAttention: topic(),
  "scaling-law": topic(),
  RLHF: topic(),
  RLVR: topic("outline"),
  "low-precision": topic("outline"),
  "unified-MLLM": topic("outline"),
  "kl-divergence": { contentType: "analysis", completion: "growing", featuredOrder: 1, summary: "围绕 KL 散度的估计与优化展开推导，辨析不同形式的含义和适用条件。" },
  train_inference_mistahch: { contentType: "analysis", completion: "growing", featuredOrder: 2, summary: "分析训练与推理不一致如何影响策略优化，梳理误差来源与修正思路。" },
};

export const collectionHref = (type: ContentType) => `/blog/collections/${type}/`;
