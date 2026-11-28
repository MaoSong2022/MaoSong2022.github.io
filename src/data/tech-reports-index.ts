export type TechReportKind = "note" | "brief";

export type TechReportEntry = {
  /** Model display name */
  name: string;
  /** Organization or team */
  org: string;
  /** Release month (YYYY-MM) */
  release: string;
  /** Quarterly digest article slug under /blog/<digest>/ */
  digest: string;
  /** Human-readable quarter label, e.g. "2026 Q4" */
  quarter: string;
  /** In-page heading anchor id inside the digest */
  anchor: string;
  /** Base model or architecture scale */
  baseOrArch: string;
  /** Non-exclusive technical tags for cross-quarter filtering */
  tags: string[];
  /** Concise one-line takeaway */
  summary: string;
  /** Full reading note vs short brief */
  kind: TechReportKind;
};

/**
 * Single source of truth for single-model technical report notes organized by
 * quarterly digests. Adding a row automatically updates:
 *   1. The in-article summary table (<ReportDigestTable />)
 *   2. The cross-quarter filterable index on /blog/collections/reports/
 *   3. Global keyword search on /browse
 */
export const TECH_REPORTS_INDEX: TechReportEntry[] = [
  {
    name: "Rufus-Air",
    org: "Amazon",
    release: "2026-09",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "rufus-air",
    baseOrArch: "GLM-4.5-Air-Base",
    tags: ["Post-training", "RLVR", "Agentic RL", "TIM", "Open Source"],
    summary: "四阶段渐进式后训练配方（SFT → RLVR → Agentic RL → RLHF），含完整数据配比、GSPO 与 R3/TIS。",
    kind: "note",
  },
  {
    name: "K2-Horizon",
    org: "IFM",
    release: "2026-11",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "k2-horizon",
    baseOrArch: "0.9B–375B-A23B (Dense/MoE)",
    tags: ["Pre-training", "MoE", "Architecture", "Long Context", "Open Source"],
    summary: "6 尺寸开源模型族；提出 Grouped RMSNorm、MoVA、Softplus Gating 与扩散蒸馏投机解码 Uno。",
    kind: "note",
  },
  {
    name: "Apex Agents",
    org: "Mercor & SkyRL",
    release: "2026-11",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "apex-agents",
    baseOrArch: "Qwen3.5 / Qwen3.6",
    tags: ["Agentic RL", "Async RL", "Post-training"],
    summary: "知识工作 Agentic RL 实战：Harbor + Modal 沙箱、TITO、DPPO + prompt-mean + context nudge 消融。",
    kind: "note",
  },
  {
    name: "FireFlash",
    org: "Agent Economics",
    release: "2026-11",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "fireflash",
    baseOrArch: "GLM-5.3 Flash",
    tags: ["Agentic RL", "Async RL", "Post-training", "TIM"],
    summary: "64×B300 上 5 天异步 RL 配方：通过率 [15%,85%] 过滤、长度奖励、TIS 与 score centering。",
    kind: "note",
  },
  {
    name: "Ripple 65B",
    org: "Ricursive Intelligence",
    release: "2026-11",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "ripple-65b",
    baseOrArch: "65B Dense (Thin & Deep)",
    tags: ["Code", "Pre-training", "Hybrid Attention", "Long Context", "TIM"],
    summary: "80 层瘦深 Dense 架构（MFU 40%），结合 SWA/GQA、Gated Attention、Global NoPE 与 Value Residual。",
    kind: "note",
  },
  {
    name: "Ultramarine 1 Flash",
    org: "Paper Instruments",
    release: "2026-10",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "ultramarine-1-flash",
    baseOrArch: "GLM-5.3 Flash",
    tags: ["Agentic RL", "Post-training", "CISPO", "TIM"],
    summary: "面向不可验证环境的 Rubric Grading 奖励框架，结合 CISPO、PipelineRL、R3 与 DAPO。",
    kind: "note",
  },
  {
    name: "IQuest-Q1",
    org: "IQuest",
    release: "2026-09",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "iquest-q1",
    baseOrArch: "320B-A15B MoE",
    tags: ["Code", "Pre-training", "MoE", "Hybrid Attention", "Architecture", "Agentic RL"],
    summary: "29T Token 预训练与中训练；SWA:GQA=3:1、Stable Split Norm、MTP 递归草稿与 4 专家 MOPD 合版。",
    kind: "note",
  },
  {
    name: "North Mini Code",
    org: "Cohere",
    release: "2026-09",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "north-mini-code",
    baseOrArch: "30B-A3B MoE",
    tags: ["Code", "MoE", "Hybrid Attention", "Async RL", "RLVR", "CISPO"],
    summary: "30B-A3B 代码 MoE：SWA:GQA=3:1，70K 可验证仓库任务 Async RLVR（CISPO + Harbor/SWE-agent）。",
    kind: "note",
  },
  {
    name: "Macaron-V1-Preview",
    org: "Mind Lab",
    release: "2026-10",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "macaron-v1-preview",
    baseOrArch: "GLM-5.1 (749B MoL)",
    tags: ["Agentic RL", "Post-training", "TIM", "Self-Evolution"],
    summary: "Mixture of LoRA（MoL）专家工具路由、跨 LoRA KV Cache 复用与基于轨迹蒸馏的 Self-Evolution 闭环。",
    kind: "note",
  },
  {
    name: "Ornith 1.0",
    org: "DeepReinforce",
    release: "2026-06",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "ornith",
    baseOrArch: "Gemma 4 / Qwen3.5 (9B–397B)",
    tags: ["Agentic RL", "Code", "Async RL", "Self-Evolution"],
    summary: "同步优化 LLM 与 Scaffold 的自演化训练，含安全边界隔离、Monitor 监督与 Staleness 加权异步 RL。",
    kind: "note",
  },
  {
    name: "Laguna M.1 / XS.2",
    org: "Poolside",
    release: "2026-05",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "laguna",
    baseOrArch: "226B-A23B / 33B-A3B MoE",
    tags: ["Code", "Post-training", "MoE", "CISPO", "Agentic RL"],
    summary: "Agentic Coding 后训练配方：special token 嵌入预热防尖峰与长度加权 LOO baseline（CISPO）。",
    kind: "note",
  },
  {
    name: "Beam",
    org: "Reflection",
    release: "2026-10",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "beam",
    baseOrArch: "501B-A23B MoE (1M ctx)",
    tags: ["Code", "MoE", "Long Context"],
    summary: "聚焦 Coding 与 Agentic 场景的 501B-A23B 开源权重 MoE，支持 1M 上下文。",
    kind: "brief",
  },
  {
    name: "Marin 535B",
    org: "Open Athena / Stanford",
    release: "2026-10",
    digest: "tech-reports-2026-q4",
    quarter: "2026 Q4",
    anchor: "marin",
    baseOrArch: "535B-A23B LatentMoE",
    tags: ["Open Source", "Pre-training", "MoE", "Hybrid Attention"],
    summary: "完全开源的大规模 MoE 训练项目，采用 SWA:GQA=2:1 与 LatentMoE（384 选 8）。",
    kind: "brief",
  },
];

export function reportAnchorHref(entry: TechReportEntry): string {
  return `/blog/${entry.digest}/#${entry.anchor}`;
}

export function getReportsByDigest(digestSlug: string): TechReportEntry[] {
  return TECH_REPORTS_INDEX.filter((r) => r.digest === digestSlug);
}

export function getAllReportTags(): string[] {
  return [...new Set(TECH_REPORTS_INDEX.flatMap((r) => r.tags))].sort((a, b) =>
    a.localeCompare(b),
  );
}

export function getAllReportQuarters(): Array<{ digest: string; quarter: string }> {
  const seen = new Map<string, string>();
  for (const r of TECH_REPORTS_INDEX) {
    if (!seen.has(r.digest)) seen.set(r.digest, r.quarter);
  }
  return Array.from(seen.entries()).map(([digest, quarter]) => ({
    digest,
    quarter,
  }));
}

/**
 * Extra search keywords injected into /browse for a quarterly digest article
 * so searching any individual model name, org, or tag finds its digest.
 */
export function getDigestSearchText(digestSlug: string): string {
  const rows = getReportsByDigest(digestSlug);
  if (!rows.length) return "";
  return rows
    .flatMap((r) => [r.name, r.org, r.baseOrArch, r.release, ...r.tags, r.summary])
    .join(" ");
}
