import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const bundled = await build({ entryPoints: ['src/utils/site-articles.ts'], bundle: true, write: false, format: 'esm', platform: 'node' });
const { parseArticleGlob, selectFeatured, sortByCompletion } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
const bundledTags = await build({ entryPoints: ['src/utils/tags.ts'], bundle: true, write: false, format: 'esm', platform: 'node' });
const { normalizeTagList, collectUniqueTags } = await import(`data:text/javascript;base64,${Buffer.from(bundledTags.outputFiles[0].text).toString('base64')}`);
const entry = (slug, published = '2026-01-01') => [`../content/${slug}/article.mdx`, { frontmatter: { title: slug, published } }];

test('types and completion are independent; scaffolding stays out of the public index', async () => {
  const rows = await parseArticleGlob(Object.fromEntries(['rl4llm', 'kimi', 'long-context', 'kl-divergence', 'rlvr', 'demo'].map(s => entry(s))));
  assert.equal(rows.length, 5);
  const bySlug = Object.fromEntries(rows.map(r => [r.slug, r]));
  assert.equal(bySlug.rl4llm.contentType, 'learning');
  assert.equal(bySlug.rl4llm.completion, 'complete');
  assert.equal(bySlug.kimi.contentType, 'reports');
  assert.equal(bySlug['long-context'].contentType, 'topics');
  assert.equal(bySlug['kl-divergence'].contentType, 'analysis');
  assert.equal(bySlug['kl-divergence'].completion, 'growing');
  assert.equal(bySlug.rlvr.completion, 'outline');
});

test('fresh drafts cannot displace curated tutorials and analyses', async () => {
  const rows = await parseArticleGlob(Object.fromEntries([
    entry('deep-learning', '2099-01-01'), entry('kimi'), entry('tokenizer'),
    entry('rl4llm'), entry('llm-resource-computation'), entry('kl-divergence'), entry('train-inference-mismatch'),
  ]));
  assert.deepEqual(selectFeatured(rows, 'learning').map(r => r.slug), ['rl4llm', 'llm-resource-computation', 'tokenizer']);
  assert.deepEqual(selectFeatured(rows, 'analysis').map(r => r.slug), ['kl-divergence', 'train-inference-mismatch']);
});

test('lists prioritize completion, then retain update order, without mutating input', () => {
  const rows = [{ slug: 'new-draft', completion: 'outline' }, { slug: 'new-complete', completion: 'complete' }, { slug: 'growing', completion: 'growing' }, { slug: 'old-complete', completion: 'complete' }];
  assert.deepEqual(sortByCompletion(rows).map(r => r.slug), ['new-complete', 'old-complete', 'growing', 'new-draft']);
  assert.equal(rows[0].slug, 'new-draft');
});

test('every existing article has an explicit classification; new content defaults conservatively', async () => {
  const dirs = await readdir('src/content', { withFileTypes: true });
  const slugs = [];
  for (const dir of dirs.filter(d => d.isDirectory())) {
    const files = await readdir(`src/content/${dir.name}`);
    if (files.some(f => /^article\.mdx?$/.test(f))) slugs.push(dir.name);
  }
  const rows = await parseArticleGlob(Object.fromEntries(slugs.map(s => entry(s))));
  assert.deepEqual(rows.map(r => r.slug).sort(), slugs.filter(s => s !== 'demo').sort());
  assert.ok(rows.every(r => r.classified), 'all existing articles need editorial review');
  const [fresh] = await parseArticleGlob(Object.fromEntries([entry('new-unreviewed-post')]));
  assert.equal(fresh.completion, 'outline');
  assert.equal(fresh.classified, false);
  assert.deepEqual(selectFeatured([fresh], 'learning'), []);
});

test('all content folder names use lowercase kebab-case', async () => {
  const dirs = await readdir('src/content', { withFileTypes: true });
  const invalid = dirs
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .filter((name) => !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name));
  assert.deepEqual(invalid, [], `Non-kebab-case content folders: ${invalid.join(', ')}`);
});

test('tag normalization deduplicates case-insensitively and collapses whitespace', () => {
  assert.deepEqual(
    normalizeTagList(['  Attention ', 'attention', 'Long  Context', 'MoE', 'moe', '']),
    ['Attention', 'Long Context', 'MoE'],
  );
  assert.deepEqual(
    collectUniqueTags([['RL', 'Attention'], ['attention', 'MoE'], ['rl', 'Agentic RL']]),
    ['Agentic RL', 'Attention', 'MoE', 'RL'],
  );
});

test('all article and tech-report frontmatter tags follow unified format without duplicates', async () => {
  const dirs = await readdir('src/content', { withFileTypes: true });
  const targetFiles = [];
  for (const dir of dirs.filter((d) => d.isDirectory())) {
    if (dir.name === 'tech-reports') {
      const reports = await readdir('src/content/tech-reports');
      for (const f of reports) {
        if (/\.mdx?$/.test(f)) targetFiles.push(`src/content/tech-reports/${f}`);
      }
      continue;
    }
    const files = await readdir(`src/content/${dir.name}`);
    const article = files.find((f) => /^article\.mdx?$/.test(f));
    if (article) targetFiles.push(`src/content/${dir.name}/${article}`);
  }

  const BANNED_META_TAGS = new Set(['tutorial', 'technical report', 'rl', 'math']);
  const ALLOWED_PUBLIC_SUBJECTS = new Set(['LLM', 'MLLM', 'RL', 'Infra', 'Math']);
  const seenCanonical = new Map();
  const issues = [];

  for (const file of targetFiles.sort()) {
    const raw = await readFile(file, 'utf8');
    const fmMatch = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
    if (!fmMatch) {
      issues.push(`${file}: missing frontmatter`);
      continue;
    }
    const fm = fmMatch[1];
    const isDraft = /^draft:\s*true$/m.test(fm);
    const isDemo = file === 'src/content/demo/article.mdx';
    const isReport = file.startsWith('src/content/tech-reports/');
    if (!isDraft && !isDemo && !isReport) {
      const catMatch = fm.match(/^categories:[^\r\n]*((?:\r?\n[ \t]*-[^\r\n]*)*)/m);
      const firstCat = catMatch
        ? catMatch[1]
            .split(/\r?\n/)
            .map((l) => l.replace(/^[ \t]*-\s*/, '').replace(/^["']|["']$/g, '').trim())
            .find(Boolean)
        : undefined;
      if (!firstCat || !ALLOWED_PUBLIC_SUBJECTS.has(firstCat)) {
        issues.push(`${file}: primary subject must be one of ${[...ALLOWED_PUBLIC_SUBJECTS].join(', ')}, got ${JSON.stringify(firstCat)}`);
      }
    }
    const tagMatch = fm.match(/^tags:([^\r\n]*)((?:\r?\n[ \t]+-[^\r\n]*)*)/m);
    if (!tagMatch) {
      if (!isDraft) issues.push(`${file}: missing tags`);
      continue;
    }
    if (tagMatch[1].trim() !== '') {
      issues.push(`${file}: tags must use block list syntax (not inline "${tagMatch[1].trim()}")`);
      continue;
    }
    const itemLines = tagMatch[2]
      .split(/\r?\n/)
      .filter((line) => line.length > 0);
    if (itemLines.length === 0 && !isDraft) {
      issues.push(`${file}: empty tags list`);
      continue;
    }

    const fileSeen = new Set();
    for (const line of itemLines) {
      const m = line.match(/^  - "([^"]+)"$/);
      if (!m) {
        issues.push(`${file}: tag item must use two-space double-quoted format, got ${JSON.stringify(line)}`);
        continue;
      }
      const tag = m[1];
      const lower = tag.toLowerCase();
      if (tag !== tag.trim() || /\s{2,}/.test(tag)) {
        issues.push(`${file}: tag "${tag}" has irregular whitespace`);
      }
      if (!/^[A-Z0-9]/.test(tag)) {
        issues.push(`${file}: tag "${tag}" must start with an uppercase letter or digit`);
      }
      if (BANNED_META_TAGS.has(lower)) {
        issues.push(`${file}: banned meta tag "${tag}"`);
      }
      if (fileSeen.has(lower)) {
        issues.push(`${file}: duplicate tag "${tag}" within file`);
      }
      fileSeen.add(lower);

      const prev = seenCanonical.get(lower);
      if (prev && prev.tag !== tag) {
        issues.push(`${file}: inconsistent tag casing "${tag}" vs "${prev.tag}" in ${prev.file}`);
      } else if (!prev) {
        seenCanonical.set(lower, { tag, file });
      }
    }
  }

  assert.deepEqual(issues, [], `Tag validation issues:\n${issues.join('\n')}`);
});
