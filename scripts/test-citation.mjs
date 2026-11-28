import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: ['src/utils/citation.ts'],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {
  buildArticleCitation,
  extractCitationDate,
  formatCitationAuthor,
} = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
);

test('formats Mao Song in academic Last, First order and includes URL, month, and host', () => {
  const result = buildArticleCitation({
    title: 'MoE 架构详解：路由机制、负载均衡与大规模训练设计',
    authors: ['Mao Song'],
    published: '2025-12-13 16:04:04+0800',
    url: 'https://maosong.website/blog/moe/',
    slug: 'moe',
  });

  assert.equal(
    result.citationText,
    'Mao, Song. (Dec 2025). "MoE 架构详解：路由机制、负载均衡与大规模训练设计". maosong.website. https://maosong.website/blog/moe/',
  );
  assert.equal(result.bibKey, 'mao2025moe');
  assert.equal(
    result.bibtex,
    [
      '@misc{mao2025moe,',
      '  title        = {{MoE 架构详解：路由机制、负载均衡与大规模训练设计}},',
      '  author       = {Mao, Song},',
      '  year         = {2025},',
      '  month        = {Dec},',
      '  howpublished = {maosong.website},',
      '  url          = {https://maosong.website/blog/moe/}',
      '}',
    ].join('\n'),
  );
});

test('uses slug for BibTeX key so pure-Chinese titles never produce broken keys', () => {
  const result = buildArticleCitation({
    title: '随机游走：二维和三维的醉汉能否回家？',
    authors: ['Mao Song'],
    published: '2026-01-24 16:32:14+0800',
    url: 'https://maosong.website/blog/random-walk/',
    slug: 'random-walk',
  });

  assert.equal(result.bibKey, 'mao2026random_walk');
  assert.match(result.bibtex, /^@misc\{mao2026random_walk,/);
});

test('parses diverse frontmatter date formats consistently in Asia/Shanghai timezone', () => {
  assert.deepEqual(extractCitationDate('Sep. 01, 2025'), { year: 2025, month: 'Sep' });
  assert.deepEqual(extractCitationDate('May. 14, 2026'), { year: 2026, month: 'May' });
  assert.deepEqual(extractCitationDate('2026-11-26'), { year: 2026, month: 'Nov' });
  assert.deepEqual(extractCitationDate('2026-11'), { year: 2026, month: 'Nov' });
  assert.deepEqual(extractCitationDate('2025'), { year: 2025 });
  assert.deepEqual(extractCitationDate(new Date('2026-01-24T08:32:14.000Z')), {
    year: 2026,
    month: 'Jan',
  });
});

test('handles report slugs, DOI, and non-site or multi-author names gracefully', () => {
  assert.equal(formatCitationAuthor('Mao Song'), 'Mao, Song');
  assert.equal(formatCitationAuthor('Mao, Song'), 'Mao, Song');
  assert.equal(formatCitationAuthor('Hugging Face'), 'Hugging Face');

  const result = buildArticleCitation({
    title: 'Apex Agents',
    authors: ['Mao Song', 'Jane Doe'],
    published: '2026-11-16',
    url: 'https://maosong.website/blog/reports/apex-agents/',
    slug: 'reports/apex-agents',
    doi: '10.1234/example',
  });

  assert.equal(result.bibKey, 'mao2026reports_apex_agents');
  assert.equal(
    result.citationText,
    'Mao, Song; Jane Doe. (Nov 2026). "Apex Agents". maosong.website. https://maosong.website/blog/reports/apex-agents/ DOI: https://doi.org/10.1234/example',
  );
  assert.match(result.bibtex, /author\s+=\s+\{Mao, Song and Jane Doe\}/);
  assert.match(result.bibtex, /doi\s+=\s+\{10\.1234\/example\}/);
});

test('omits date cleanly without double punctuation when published is missing', () => {
  const result = buildArticleCitation({
    title: 'Draft Note',
    authors: ['Mao Song'],
    url: 'https://maosong.website/blog/draft-note/',
    slug: 'draft-note',
  });

  assert.equal(
    result.citationText,
    'Mao, Song. "Draft Note". maosong.website. https://maosong.website/blog/draft-note/',
  );
  assert.equal(result.bibKey, 'mao_draft_note');
  assert.doesNotMatch(result.bibtex, /year\s*=/);
  assert.doesNotMatch(result.bibtex, /month\s*=/);
});
