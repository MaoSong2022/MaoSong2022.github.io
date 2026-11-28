# 文章分类与完成度维护

首页、文章总览、四类独立入口和搜索页共用 `src/data/article-editorial.ts`。分类不改动原有文章 URL；技术主题（categories）和标签（tags）仍从文章 frontmatter 读取。

## 两个独立维度

| contentType | 用途 | 独立入口 |
| --- | --- | --- |
| learning | 从零建立基础的学习笔记与教程 | /blog/collections/learning/ |
| reports | 模型或团队技术报告的阅读记录 | /blog/collections/reports/ |
| topics | 一个技术方向的持续追踪 | /blog/collections/topics/ |
| analysis | 围绕具体问题的推导、分析与判断 | /blog/collections/analysis/ |

| completion | 页面标记 | 判断标准 |
| --- | --- | --- |
| complete | Ready to Read | 当前主题已有连贯讲解，可按篇阅读；仍可继续更新 |
| growing | In Progress | 已有主体内容，部分章节、推导或验证仍在补充 |
| outline | Outline | 以目录、材料或初步记录为主 |

完成度是编辑判断，不按字数或更新时间推断，也不使用百分比。技术报告和方向追踪不是天然低完成度；当其声明的范围得到完整覆盖时也可以标为 complete。

## 新增或更新文章

1. 在 `ARTICLE_EDITORIAL` 中添加对应 slug 的 `contentType` 和 `completion`。未登记的新文章会保守地显示为学习笔记 / 提纲，测试会提示补全分类。
2. 达到所声明主题的完整阅读要求后更新 completion。类别发生变化时只需更新 contentType。
3. 要在首页推荐，显式设置 `featuredOrder`，数字越小越靠前。首页教程必须是 learning + complete；深度分析必须是 analysis 且不是 outline。仅更新文章日期不会自动成为精选。
4. 可添加 `summary` 提供面向读者的短摘要；未设置时使用 frontmatter.description。
5. 运行 `node --test scripts/test-article-index.mjs` 和 `npm run build`。

列表每类先按完成度排序，同完成度保留更新时间倒序。搜索支持 `type`、`completion`、`q`、`category` 和多个 `tag` 参数，可分享筛选结果，例如 `/browse?type=learning&completion=complete`。

## Subjects 与 Tags 统一格式规范

1. **Subjects（`categories`）**：主学科统一归入 5 个公开主题：`LLM`、`MLLM`、`RL`、`Infra`、`Math`（草稿可暂用 `Deep Learning`）。`RL` 与 `Math` 作为 Subject 管理，不再重复挂入 `tags`。
2. **YAML 格式**：`tags` 统一使用双引号多行列表格式，不混用无引号或行内数组：
   ```yaml
   tags:
     - "MoE"
     - "Pre-training"
   ```
3. **命名风格**：统一使用 **Title Case + 保留专有缩写与官方写法**，例如 `Attention`、`Position Encoding`、`Long Context`、`Hybrid Attention`、`Pre-training`、`Post-training`、`Agentic RL`、`Async RL`、`MoE`、`MLLM`、`TIM`、`CS336`。
4. **模型系列标签**：模型系列技术报告文章使用**模型系列名**而非厂商名（如 `Qwen`、`DeepSeek`、`Gemini`、`Gemma`、`GLM`、`GPT`、`Hunyuan`、`InternVL`、`KAT-Coder`、`Keye-VL`、`Kimi`、`LLaMA`、`MAI`、`MiMo`、`MiniCPM`、`MiniMax`、`Mistral`、`Nemotron`、`OLMo`、`Step`、`Apertus`、`Composer`）；通用技术综述不挂载零散模型名标签。
5. **粒度控制与去重**：避免过于特化、仅服务单篇子主题的标签（如 `KL Divergence`、`DPO`、`Random Walk`、`Sampling`、`Roofline`）及与 `contentType` 重复的元标签（如 `tutorial`、`Technical Report`）；全站与单篇内均不出现仅大小写不同的重复标签（由 `src/utils/tags.ts` 与 `scripts/test-article-index.mjs` 共同约束）。

模板文章 demo 不出现在公共文章索引和 RSS 中，原有直接访问路径保留。其他提纲仍在对应分类页和搜索中可见。
