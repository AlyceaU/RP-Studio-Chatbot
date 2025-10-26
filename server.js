// Simple server (CommonJS) that serves chat.html at http://localhost:3000
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files & chat page
app.use(express.static(__dirname));
app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'chat.html')));

// --- Load extra instructions if you have them ---
let instructions = '';
try {
  instructions = fs.readFileSync(path.join(__dirname, 'RPTT_INSTRUCTIONS.txt'), 'utf8');
} catch (_) { /* optional file */ }

// --- pdf-parse import (works with both CJS and ESM builds) ---
let pdfParse = null;
try {
  pdfParse = require('pdf-parse');
  if (pdfParse && typeof pdfParse !== 'function' && pdfParse.default) {
    pdfParse = pdfParse.default;
  }
} catch (e) {
  console.warn('pdf-parse not installed. PDFs will be skipped.', e.message);
}

// --- Simple knowledge loader: reads .txt and .pdf from ./knowledge ---
const KNOWLEDGE_DIR = path.join(__dirname, 'knowledge');
let knowledgeChunks = [];

async function loadKnowledge() {
  knowledgeChunks = [];
  if (!fs.existsSync(KNOWLEDGE_DIR)) {
    console.log('knowledge/ folder not found—continuing without local knowledge.');
    return;
  }
  const files = fs.readdirSync(KNOWLEDGE_DIR);

  for (const f of files) {
    const full = path.join(KNOWLEDGE_DIR, f);
    const stat = fs.statSync(full);
    if (!stat.isFile()) continue;

    const lower = f.toLowerCase();
    try {
      if (lower.endsWith('.txt')) {
        const text = fs.readFileSync(full, 'utf8');
        pushChunks(text);
      } else if (lower.endsWith('.pdf')) {
        if (!pdfParse) {
          console.warn(`Skipping PDF "${f}" because pdf-parse is unavailable.`);
          continue;
        }
        try {
          const data = fs.readFileSync(full);
          const parsed = await pdfParse(data);
          pushChunks(parsed.text || '');
        } catch (e) {
          console.warn(`PDF parse failed for ${f}: ${e.message}`);
        }
      }
    } catch (e) {
      console.warn(`Failed reading ${f}: ${e.message}`);
    }
  }
  console.log(`Loaded ${knowledgeChunks.length} chunks from knowledge/`);
}

function pushChunks(text) {
  if (!text) return;
  // Split on blank lines and trim
  const parts = text
    .split(/\n{2,}/)
    .map(s => s.trim())
    .filter(Boolean);

  for (const p of parts) {
    if (p.length > 2000) {
      // Soft chunking for long paragraphs
      for (let i = 0; i < p.length; i += 2000) {
        knowledgeChunks.push(p.slice(i, i + 2000));
      }
    } else {
      knowledgeChunks.push(p);
    }
  }
}

// --- Smarter keyword retriever with synonyms, phrase boosts, and variant handling ---
function findRelevantChunks(question, max = 5) {
  if (!knowledgeChunks.length) return [];

  const norm = s => (s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const q = norm(question);

  const synonymSets = [
    ['retest', 're test', 're-test', 'retake', 're take', 're-take'],
    ['fee', 'cost', 'charge', 'payment', 'price'],
    ['wait', 'waiting', 'delay', 'hold', 'gap', 'cooldown', 'cool down'],
    ['period', 'window', 'timeframe', 'timeline'],
    ['hours', 'requirements', 'prerequisites', 'prereq'],
    ['policy', 'rule', 'guideline', 'guidance'],
    ['exam', 'practical', 'test', 'assessment'],
    ['schedule', 'book', 'arrange'],
  ];

  const qTokens = new Set(q.split(' ').filter(w => w.length > 2));
  for (const set of synonymSets) {
    if (set.some(w => qTokens.has(w))) {
      for (const w of set) qTokens.add(w);
    }
  }

  const phraseBoosts = [
    'retest fee',
    're test fee',
    're-test fee',
    'waiting period',
    'wait period',
    'retest waiting',
    're test waiting',
    're-test waiting',
    'how many hours',
    'hours to retest',
    'hours before retest',
  ];

  const scored = knowledgeChunks.map(c => {
    const text = norm(c);
    let score = 0;

    for (const t of qTokens) {
      if (text.includes(t)) score += 1;
    }
    for (const p of phraseBoosts) {
      if (text.includes(p)) score += 6;
    }
    if (/\bre[\s-]?test\b/.test(q) && /\bre[\s-]?test\b/.test(text)) score += 4;

    const hasNumberInQ = /\b\d+/.test(q);
    const hasNumberInC = /\b\d+/.test(text);
    if (hasNumberInQ && hasNumberInC) score += 2;

    return { c, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, max)
    .filter(x => x.score > 0)
    .map(x => x.c);
}

// --- OpenAI client ---
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// --- Chat route (system is DEFINED INSIDE here) ---
app.post('/api/chat', async (req, res) => {
  try {
    console.log('POST /api/chat body:', req.body);
    const userMessage = (req.body && req.body.message) ? String(req.body.message) : '';

    const hits = findRelevantChunks(userMessage, 5);
    const context =
      hits.length
        ? `\n\nStudio Reference (from uploaded PDFs):\n${hits.map((c, i) => `[${i + 1}] ${c}`).join('\n\n')}\n\n`
        : '';

    const system = `You are the Real Pilates® Studio Team Assistant.
Use warm, professional, on-brand language aligned with Real Pilates® studio operations.
ONLY answer using the Studio policies and the provided "Studio Reference" context below.
If the answer is not clearly present, say: "I don’t have that in the RP Studio docs."${instructions ? '\n\n' + instructions : ''}`;

    const out = await client.responses.create({
      model: 'gpt-4o-mini',
      input: [
        { role: 'system', content: system + context },
        { role: 'user',   content: userMessage }
      ]
    });

    // Robust extraction of text across SDK versions
    let reply = '';
    if (out.output_text) {
      reply = out.output_text;
    } else if (Array.isArray(out.output)) {
      // Newer SDK: iterate events
      reply = out.output
        .flatMap(ev => (ev?.content || []))
        .map(c => c?.text?.value || '')
        .join('\n')
        .trim();
    }
    if (!reply && out?.response?.output_text) {
      reply = out.response.output_text;
    }
    reply = reply || 'I don’t have that in the RPTT Guide.';

    console.log('reply length:', reply.length);
    res.json({ reply });
  } catch (e) {
    console.error('SERVER ERROR:', e.stack || e.message);
    res.status(500).json({ error: e && e.message ? e.message : 'Server error' });
  }
});

// --- Debug + hot reload ---
app.get('/debug', (_req, res) => {
  try {
    const dir = path.join(__dirname, 'knowledge');
    let files = [];
    let exists = false;
    try {
      exists = fs.existsSync(dir);
      if (exists) files = fs.readdirSync(dir);
    } catch (_) {}
    res.json({
      knowledgeDir: dir,
      exists,
      files,
      chunkCount: Array.isArray(knowledgeChunks) ? knowledgeChunks.length : 0
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/reload', async (_req, res) => {
  try {
    await loadKnowledge();
    res.json({ ok: true, chunkCount: knowledgeChunks.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3000;
loadKnowledge()
  .then(() => {
    app.listen(PORT, () => console.log(`RP chat server running on port ${PORT}`));
  })
  .catch(err => {
    console.error('Knowledge load failed:', err);
    app.listen(PORT, () => console.log(`RP chat server running on port ${PORT} (without knowledge)`));
  });
