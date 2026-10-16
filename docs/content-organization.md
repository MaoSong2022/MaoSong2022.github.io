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

模板文章 demo 不出现在公共文章索引和 RSS 中，原有直接访问路径保留。其他提纲仍在对应分类页和搜索中可见。
