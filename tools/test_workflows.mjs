// Offline checks for the generated n8n workflows. No n8n, no API keys, no network.
//   node tools/test_workflows.mjs
// 1) structural checks on every workflow JSON (valid graph, known node types, no dangling links)
// 2) runs the real JavaScript of each Code node against stubbed n8n globals
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const load = f => JSON.parse(fs.readFileSync(path.join(root, 'n8n', f), 'utf8'));
let failures = 0;
const ok = (name) => console.log('  ok   ' + name);
const bad = (name, e) => { failures++; console.log('  FAIL ' + name + '\n       ' + (e && e.message || e)); };
const check = (name, fn) => { try { fn(); ok(name); } catch (e) { bad(name, e); } };

// ---------------------------------------------------------------- structure
const files = ['01_ingest_f1_knowledge_base.json', '02_f1_llm_chatbot.json', '03_f1_rag_chatbot.json'];
const KNOWN = new Set([
  'n8n-nodes-base.webhook', 'n8n-nodes-base.respondToWebhook', 'n8n-nodes-base.code',
  'n8n-nodes-base.manualTrigger', 'n8n-nodes-base.stickyNote',
  '@n8n/n8n-nodes-langchain.chainLlm', '@n8n/n8n-nodes-langchain.lmChatOpenAi',
  '@n8n/n8n-nodes-langchain.embeddingsOpenAi', '@n8n/n8n-nodes-langchain.vectorStoreInMemory',
  '@n8n/n8n-nodes-langchain.documentDefaultDataLoader',
]);
const AI_KINDS = new Set(['ai_languageModel', 'ai_embedding', 'ai_document']);

console.log('\nStructure');
for (const f of files) {
  const wf = load(f);
  check(f + ': graph is consistent', () => {
    const names = wf.nodes.map(n => n.name);
    assert.equal(new Set(names).size, names.length, 'duplicate node names');
    assert.equal(new Set(wf.nodes.map(n => n.id)).size, wf.nodes.length, 'duplicate node ids');
    for (const n of wf.nodes) {
      assert.ok(KNOWN.has(n.type), 'unexpected node type ' + n.type);
      assert.ok(Array.isArray(n.position) && n.position.length === 2, 'bad position on ' + n.name);
    }
    for (const [src, kinds] of Object.entries(wf.connections)) {
      assert.ok(names.includes(src), 'connection from unknown node ' + src);
      for (const [kind, outs] of Object.entries(kinds)) {
        for (const list of outs) for (const l of list) {
          assert.ok(names.includes(l.node), `${src} -> unknown node ${l.node}`);
          assert.equal(l.type, kind, `${src}: link type ${l.type} != ${kind}`);
        }
      }
    }
    // every sub-node (model / embeddings / loader) must be attached to a root node
    const connectedSources = new Set(Object.keys(wf.connections));
    for (const n of wf.nodes) {
      if (/lmChat|embeddings|documentDefault/.test(n.type)) assert.ok(connectedSources.has(n.name), n.name + ' is not connected');
    }
    // every non-sticky, non-trigger root node must be reachable via main links
    const roots = wf.nodes.filter(n => !/stickyNote|lmChat|embeddings|documentDefault/.test(n.type)).map(n => n.name);
    const incoming = new Set();
    for (const kinds of Object.values(wf.connections)) for (const [k, outs] of Object.entries(kinds)) if (k === 'main') for (const l of outs.flat()) incoming.add(l.node);
    const triggers = wf.nodes.filter(n => /webhook$|manualTrigger/.test(n.type)).map(n => n.name);
    for (const r of roots) assert.ok(incoming.has(r) || triggers.includes(r), r + ' has no incoming link');
  });
  check(f + ': prompts contain no curly braces', () => {
    for (const n of wf.nodes.filter(n => n.type.endsWith('chainLlm'))) {
      for (const m of n.parameters.messages.messageValues) assert.ok(!/[{}]/.test(m.message), n.name + ' system prompt has { or }');
    }
  });
}

check('ingest and RAG workflows use the same memory key and embedding model', () => {
  const ing = load(files[0]), rag = load(files[2]);
  const key = w => w.nodes.find(n => n.type.endsWith('vectorStoreInMemory')).parameters.memoryKey.value;
  const emb = w => w.nodes.find(n => n.type.endsWith('embeddingsOpenAi')).parameters.model;
  assert.equal(key(ing), key(rag));
  assert.equal(emb(ing), emb(rag));
});

// ---------------------------------------------------------------- code nodes
function runCode(wf, nodeName, { items, named = {} }) {
  const n = wf.nodes.find(x => x.name === nodeName);
  assert.ok(n, 'no node ' + nodeName);
  const mk = arr => ({ first: () => arr[0], all: () => arr, item: arr[0] });
  const $input = mk(items);
  const $ = name => { if (!named[name]) throw new Error('stub missing for $(' + name + ')'); return mk(named[name]); };
  const fn = new Function('$input', '$', n.parameters.jsCode);
  return fn($input, $);
}
const J = o => ({ json: o });

const rag = load(files[2]);
const llm = load(files[1]);
const kb = JSON.parse(fs.readFileSync(path.join(root, 'n8n', 'kb_chunks.json'), 'utf8'));

console.log('\nPrepare');
check('builds live context, history and hint', () => {
  const [o] = runCode(rag, 'Prepare', { items: [J({ body: {
    question: '  Why did they slow down?  ',
    context: { event: 'Abu Dhabi GP', session: 'Race', lap: 23, status: 'Safety Car', note: 'cars bunched up' },
    history: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }],
  } })] });
  assert.equal(o.json.question, 'Why did they slow down?');
  assert.match(o.json.live, /Event: Abu Dhabi GP .* Lap: 23 .* Safety Car/);
  assert.equal(o.json.liveHint, 'Safety Car cars bunched up');
  assert.match(o.json.historyText, /^Viewer: hi\nAssistant: hello$/);
});
check('works with no context or history', () => {
  const [o] = runCode(rag, 'Prepare', { items: [J({ body: { question: 'what is DRS' } })] });
  assert.equal(o.json.live, 'not provided');
  assert.equal(o.json.historyText, '(no earlier messages)');
});
check('rejects an empty question', () => {
  assert.throws(() => runCode(rag, 'Prepare', { items: [J({ body: { question: '   ' } })] }), /question/);
});
check('truncates absurdly long input and caps history to 6 turns', () => {
  const hist = Array.from({ length: 20 }, (_, i) => ({ role: 'user', content: 'm' + i }));
  const [o] = runCode(rag, 'Prepare', { items: [J({ body: { question: 'x'.repeat(5000), history: hist } })] });
  assert.equal(o.json.question.length, 600);
  assert.equal(o.json.historyText.split('\n').length, 6);
});

console.log('\nParse sub-queries');
const prepOut = { question: 'Why did everyone slow down?', liveHint: 'Safety Car', live: 'x', historyText: '' };
check('parses numbered / quoted lines, de-duplicates, caps at 4 queries', () => {
  const out = runCode(rag, 'Parse sub-queries', {
    items: [J({ text: '1. "safety car"\n2) virtual safety car delta\n- yellow flag slow down\n- yellow flag slow down\n' })],
    named: { Prepare: [J(prepOut)] },
  });
  const qs = out.map(o => o.json.q);
  assert.equal(qs[0], 'Why did everyone slow down?');
  assert.equal(qs[1], 'Why did everyone slow down? Safety Car');
  assert.equal(qs.length, 4);
  assert.ok(qs.includes('safety car'));
});
check('falls back to the original question if the LLM call failed', () => {
  const out = runCode(rag, 'Parse sub-queries', { items: [J({ error: 'boom' })], named: { Prepare: [J({ ...prepOut, liveHint: '' })] } });
  assert.deepEqual(out.map(o => o.json.q), ['Why did everyone slow down?']);
});

console.log('\nHybrid merge: lexical (BM25) half, dense half empty (worst case)');
// Each test: question -> a chunk title that must appear in the top 5 context notes.
const tests = [
  ['What does parc fermé mean?', 'Parc fermé'],
  ['what is parc ferme', 'Parc fermé'],
  ['What is an undercut?', 'Overcut and undercut'],
  ['What does a yellow flag mean?', 'Yellow flag (single)'],
  ['what does the blue flag mean', 'Blue flag'],
  ['How many points does the winner of a sprint get?', 'Sprint race'],
  ['How many points for winning a grand prix?', 'Championship points for a grand prix'],
  ['Is DRS still used?', 'Is DRS still used in 2026'],
  ['What is overtake mode?', 'Overtake Mode'],
  ['Why do cars slow at the end of the straight?', 'Why cars seem to slow at the end of straights'],
  ['what is a virtual safety car', 'Virtual Safety Car (VSC)'],
  ['What is a stop-go penalty', 'Stop-go penalty'],
  ['how long is a 5 second penalty served', 'Time penalties (5 and 10 seconds)'],
  ['What are the tire colors?', 'Dry tyre colours'],
  ['what is dirty air', 'Dirty air and following another car'],
  ['What is the black and white flag?', 'Black and white flag (diagonal)'],
  ['How does qualifying work?', 'Qualifying format (Q1, Q2, Q3)'],
  ['What is a red flag?', 'Red flag'],
  ['Who builds the Aston Martin engine?', 'Who builds the engines in 2026'],
  ['What does purple sector mean on screen?', 'Purple, green and yellow lap times'],
  ['What is the cost cap', 'Cost cap'],
  ['what does box box mean', 'Box, Box (pit call)'],
  ['Why is Monaco a shorter race?', 'Race distance and time limit'],
  ['what is track limits', 'Track limits'],
  ['what is porpoising', 'Porpoising'],
  ['what is aquaplaning', 'Aquaplaning'],
  ['what is the pit window', 'Pit window'],
  ['what does FP1 mean', 'Common abbreviations (FP1, Q1, SQ, GP, SC, VSC)'],
  ['what is a hat trick in F1', 'Hat-trick and grand slam'],
  ['what is telemetry', 'Telemetry'],
];
let hits = 0;
for (const [q, want] of tests) {
  check(q, () => {
    const out = runCode(rag, 'Hybrid merge', { items: [J({})], named: {
      Prepare: [J({ question: q, live: 'not provided', liveHint: '', historyText: '' })],
      'Parse sub-queries': [J({ q })],
    } });
    const titles = out[0].json.retrieved.map(r => r.title);
    assert.ok(titles.includes(want), `expected "${want}" in top 5, got: ${titles.join(' | ')}`);
    assert.ok(out[0].json.retrieved.length >= 1 && out[0].json.retrieved.length <= 5);
    hits++;
  });
}
console.log(`  -> BM25 alone found the right note in the top 5 for ${hits}/${tests.length} questions`);

console.log('\nHybrid merge: fusion with dense hits');
check('dense-only hit is surfaced by RRF even with no keyword overlap', () => {
  // Vector store says "Safety Car" notes are the semantic match for a vague question.
  const target = kb.find(c => c.title === 'Safety Car (SC)');
  const q = 'why did everyone suddenly drive slowly in a line';
  const out = runCode(rag, 'Hybrid merge', { items: [J({ document: { pageContent: target.text, metadata: { id: target.id } }, score: 0.82 })], named: {
    Prepare: [J({ question: q, live: 'x', liveHint: '', historyText: '' })],
    'Parse sub-queries': [J({ q })],
  } });
  assert.ok(out[0].json.retrieved.some(r => r.id === target.id), 'dense hit missing from fused list');
  assert.equal(out[0].json.retrieved.find(r => r.id === target.id).dense, 0.82);
});
check('tolerates the empty item n8n emits when the store returns nothing', () => {
  const out = runCode(rag, 'Hybrid merge', { items: [J({})], named: {
    Prepare: [J({ question: 'qqqzzz xyzzy', live: 'x', liveHint: '', historyText: '' })],
    'Parse sub-queries': [J({ q: 'qqqzzz xyzzy' })],
  } });
  assert.match(out[0].json.contextText, /no matching notes/);
  assert.deepEqual(out[0].json.retrieved, []);
});
check('best dense score per chunk wins and ignores unknown ids', () => {
  const t = kb[0];
  const out = runCode(rag, 'Hybrid merge', { items: [
    J({ document: { metadata: { id: t.id } }, score: 0.3 }),
    J({ document: { metadata: { id: t.id } }, score: 0.9 }),
    J({ document: { metadata: { id: 'does-not-exist' } }, score: 1 }),
  ], named: {
    Prepare: [J({ question: 'zzz', live: 'x', liveHint: '', historyText: '' })],
    'Parse sub-queries': [J({ q: 'zzz' })],
  } });
  assert.equal(out[0].json.retrieved.length, 1);
  assert.equal(out[0].json.retrieved[0].dense, 0.9);
});

console.log('\nFormat response');
check('RAG: flags which notes the model actually cited', () => {
  const merged = { queries: ['q'], debug: {}, retrieved: [
    { id: 'a', title: 'A', category: 'C', rrf: 0.03, dense: 0.8, bm25: 3 },
    { id: 'b', title: 'B', category: 'C', rrf: 0.02, dense: null, bm25: 2 },
  ] };
  const [o] = runCode(rag, 'Format response', { items: [J({ text: 'It slows cars [2] and bunches them [2].' })], named: { 'Hybrid merge': [J(merged)] } });
  assert.equal(o.json.answer, 'It slows cars [2] and bunches them [2].');
  assert.deepEqual(o.json.sources.map(s => s.cited), [false, true]);
  assert.equal(o.json.mode, 'rag');
});
check('RAG and LLM: a failed model call still returns a friendly answer', () => {
  const merged = { queries: ['q'], debug: {}, retrieved: [] };
  const [r] = runCode(rag, 'Format response', { items: [J({ error: 'x' })], named: { 'Hybrid merge': [J(merged)] } });
  assert.equal(r.json.ok, false);
  assert.match(r.json.answer, /try again/);
  const [l] = runCode(llm, 'Format response', { items: [J({})] });
  assert.equal(l.json.ok, false);
});

console.log('\nIngest');
check('Load KB chunks emits one item per chunk with id/title/category/text', () => {
  const out = runCode(load(files[0]), 'Load KB chunks', { items: [J({})] });
  assert.equal(out.length, kb.length);
  for (const o of out) for (const k of ['id', 'title', 'category', 'text']) assert.ok(o.json[k], 'missing ' + k);
  assert.equal(new Set(out.map(o => o.json.id)).size, out.length, 'ids not unique');
});

console.log(failures ? `\n${failures} FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
