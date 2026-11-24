/** Editorial purpose and reading readiness are intentionally separate. */
export const CONTENT_TYPES = {
  learning: { label: "Learning Notes & Tutorials", shortLabel: "Learning Notes", description: "数学、算法与训练系统的基础教程和学习笔记。" },
  reports: { label: "Technical Reports", shortLabel: "Technical Reports", description: "大模型技术报告阅读笔记，关注架构与训练细节。" },
  topics: { label: "Topic Overviews", shortLabel: "Topic Overviews", description: "按技术方向整理的专题综述，关注核心工作与演进脉络。" },
  analysis: { label: "Deep Dives", shortLabel: "Deep Dives", description: "具体技术问题的深入推导与分析。" },
} as const;

export const COMPLETION = {
  complete: { label: "Ready to Read", description: "当前覆盖范围完整、逻辑连贯，适合从头到尾系统阅读。" },
  growing: { label: "In Progress", description: "已包含核心内容与主要章节，部分推导、实验或章节仍在持续补充。" },
  outline: { label: "Outline", description: "处于大纲、资料汇总或早期笔记阶段，供快速查阅方向。" },
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
const report = (completion: Completion = "growing"): Editorial => ({ contentType: "reports", completion });
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
  mistral: report(),
  olmo: report("outline"),
  "qwen-vl-series": report("outline"),
  internvl: report("outline"),
  "keye-vl": report(),
  mimo: report(),
  "LLaMA-series": report(),
  Gemma: report(),
  step: report("outline"),
  gemini: report(),
  gpt: report(),
  minicpm: report("outline"),
  tech_report: report(),
  "deepseek-r1": report("outline"),
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
  agenticRL: topic(),
  "low-precision": topic("outline"),
  "unified-MLLM": topic("outline"),
  "kl-divergence": { contentType: "analysis", completion: "growing", featuredOrder: 1, summary: "围绕 KL 散度的估计与优化展开推导，辨析不同形式的含义和适用条件。" },
  "train-inference-mismatch": { contentType: "analysis", completion: "growing", featuredOrder: 2, summary: "分析训练与推理不一致如何影响策略优化，梳理误差来源与修正思路。" },
};

export const collectionHref = (type: ContentType) => `/blog/collections/${type}/`;
