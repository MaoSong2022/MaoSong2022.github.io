import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir } from 'node:fs/promises';
import { build } from 'esbuild';

const bundled = await build({ entryPoints: ['src/utils/site-articles.ts'], bundle: true, write: false, format: 'esm', platform: 'node' });
const { parseArticleGlob, selectFeatured, sortByCompletion } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
const entry = (slug, published = '2026-01-01') => [`../content/${slug}/article.mdx`, { frontmatter: { title: slug, published } }];

test('types and completion are independent; scaffolding stays out of the public index', async () => {
  const rows = await parseArticleGlob(Object.fromEntries(['RL4LLM', 'kimi', 'long-context', 'kl-divergence', 'RLVR', 'demo'].map(s => entry(s))));
  assert.equal(rows.length, 5);
  const bySlug = Object.fromEntries(rows.map(r => [r.slug, r]));
  assert.equal(bySlug.RL4LLM.contentType, 'learning');
  assert.equal(bySlug.RL4LLM.completion, 'complete');
  assert.equal(bySlug.kimi.contentType, 'reports');
  assert.equal(bySlug['long-context'].contentType, 'topics');
  assert.equal(bySlug['kl-divergence'].contentType, 'analysis');
  assert.equal(bySlug['kl-divergence'].completion, 'growing');
  assert.equal(bySlug.RLVR.completion, 'outline');
});

test('fresh drafts cannot displace curated tutorials and analyses', async () => {
  const rows = await parseArticleGlob(Object.fromEntries([
    entry('deep_learning', '2099-01-01'), entry('kimi'), entry('tokenizer'),
    entry('RL4LLM'), entry('llm-resource-computation'), entry('kl-divergence'), entry('train_inference_mistahch'),
  ]));
  assert.deepEqual(selectFeatured(rows, 'learning').map(r => r.slug), ['RL4LLM', 'llm-resource-computation', 'tokenizer']);
  assert.deepEqual(selectFeatured(rows, 'analysis').map(r => r.slug), ['kl-divergence', 'train_inference_mistahch']);
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
