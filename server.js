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

// ---------- Knowledge loader with embeddings (TXT/PDF) ----------
const KNOWLEDGE_DIR = path.join(__dirname, 'knowledge');
const INDEX_PATH = path.join(KNOWLEDGE_DIR, '.index.json'); // cache to disk
let knowledge = [];   // [{ text, source }]
let embeddings = [];  // [[...vector...]]

let pdfParse = null;
try {
  pdfParse = require('pdf-parse');
  if (pdfParse && typeof pdfParse !== 'function' && pdfParse.default) {
    pdfParse = pdfParse.default;
  }
} catch (_) {
  console.warn('pdf-parse not installed. PDFs will be skipped.');
}

function normName(f) {
  return f.replace(/\.(pdf|txt)$/i, '').trim();
}

async function loadKnowledge() {
  knowledge = [];
  embeddings = [];

  if (!fs.existsSync(KNOWLEDGE_DIR)) {
    console.log('knowledge/ folder not found—continuing without local knowledge.');
    return;
  }
  const files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => !f.startsWith('.'));

  for (const f of files) {
    const full = path.join(KNOWLEDGE_DIR, f);
    if (!fs.statSync(full).isFile()) continue;

    const lower = f.toLowerCase();
    try {
      if (lower.endsWith('.txt')) {
        const text = fs.readFileSync(full, 'utf8');
        pushChunks(text, normName(f));
      } else if (lower.endsWith('.pdf')) {
        if (!pdfParse) { console.warn(`Skipping PDF "${f}" (no pdf-parse).`); continue; }
        try {
          const data = fs.readFileSync(full);
          const parsed = await pdfParse(data);
          const text = parsed.text || '';
          console.log(`[KB] Parsed ${f}: ${text.length} chars`);
          pushChunks(text, normName(f));
        } catch (e) {
          console.warn(`PDF parse failed for ${f}: ${e.message}`);
        }
      }
    } catch (e) {
      console.warn(`Failed reading ${f}: ${e.message}`);
    }
  }

  console.log(`Chunking complete: ${knowledge.length} chunks. Building/loading embeddings…`);
  await buildOrLoadEmbeddings();
  console.log(`Embeddings ready for ${embeddings.length} chunks.`);
}

function pushChunks(text, source) {
  if (!text) return;
  // normalize and split on blank lines
  const blocks = text.replace(/\r/g, '').split(/\n{2,}/).map(s => s.trim()).filter(Boolean);

  const MAX = 800;     // target chunk size
  const OVERLAP = 150; // keep context overlap

  for (const b of blocks) {
    if (b.length <= MAX) {
      knowledge.push({ text: b, source });
    } else {
      let i = 0;
      while (i < b.length) {
        const slice = b.slice(i, i + MAX);
        knowledge.push({ text: slice, source });
        if (i + MAX >= b.length) break;
        i += (MAX - OVERLAP);
      }
    }
  }
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function buildOrLoadEmbeddings() {
  // try loading cached index (must match chunk count)
  try {
    if (fs.existsSync(INDEX_PATH)) {
      const data = JSON.parse(fs.readFileSync(INDEX_PATH, 'utf8'));
      if (Array.isArray(data.embeddings) && data.embeddings.length === knowledge.length) {
        embeddings = data.embeddings;
        return;
      }
    }
  } catch (_) {}

  // build fresh in batches
  const texts = knowledge.map(k => k.text);
  const BATCH = 64;
  const all = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const resp = await client.embeddings.create({
      model: 'text-embedding-3-small',
      input: batch,
    });
    resp.data.forEach(item => all.push(item.embedding));
  }
  embeddings = all;

  // cache to disk
  try {
    fs.writeFileSync(INDEX_PATH, JSON.stringify({ embeddings }, null, 2), 'utf8');
  } catch (e) {
    console.warn('Failed to write embedding index:', e.message);
  }
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function findRelevantChunksEmbedding(question, k = 8) {
  if (!knowledge.length || !embeddings.length) return [];
  const qEmb = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: question
  });
  const qVec = qEmb.data[0].embedding;
  const scored = embeddings
    .map((vec, i) => ({ i, s: cosineSim(qVec, vec) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .map(({ i }) => knowledge[i]);
  return scored;
}

// ---------- Chat route (uses embeddings + concise answers) ----------
app.post('/api/chat', async (req, res) => {
  try {
    const userMessage = (req.body && req.body.message) ? String(req.body.message) : '';

    const hits = await findRelevantChunksEmbedding(userMessage, 8);
    const context =
      hits.length
        ? `\n\nStudio Reference:\n${hits.map((h, i) => `[${i + 1}] (${h.source}) ${h.text}`).join('\n\n')}\n\n`
        : '';

    const system = `You are the Real Pilates® Studio Team Assistant.
Use warm, professional, concise language (2–4 sentences).
Answer ONLY from "Studio Reference" below. If the answer is not clearly present, say "I don’t have that in the RP Studio docs."
Cite the primary source in parentheses at the end, e.g., "(Employee Handbook)".`;

    const out = await client.responses.create({
      model: 'gpt-4o-mini',
      input: [
        { role: 'system', content: system + context },
        { role: 'user',   content: userMessage }
      ]
    });

    let reply = '';
    if (out.output_text) {
      reply = out.output_text;
    } else if (Array.isArray(out.output)) {
      reply = out.output.flatMap(ev => (ev?.content || []))
        .map(c => c?.text?.value || '')
        .join('\n').trim();
    }
    reply = reply || 'I don’t have that in the RP Studio docs.';

    res.json({ reply });
  } catch (e) {
    console.error('SERVER ERROR:', e.stack || e.message);
    res.status(500).json({ error: e.message || 'Server error' });
  }
});

// ---------- Debug + reload ----------
app.get('/debug', (_req, res) => {
  try {
    let files = [];
    let exists = false;
    try {
      exists = fs.existsSync(KNOWLEDGE_DIR);
      if (exists) files = fs.readdirSync(KNOWLEDGE_DIR).filter(f => !f.startsWith('.'));
    } catch (_) {}
    res.json({
      knowledgeDir: KNOWLEDGE_DIR,
      exists,
      files,
      chunkCount: knowledge.length
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/reload', async (_req, res) => {
  try {
    await loadKnowledge();
    res.json({ ok: true, chunkCount: knowledge.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
loadKnowledge()
  .then(() => {
    app.listen(PORT, () => console.log(`RP chat server running on port ${PORT}`));
  })
  .catch(err => {
    console.error('Knowledge load failed:', err);
    app.listen(PORT, () => console.log(`RP chat server running on port ${PORT} (without knowledge)`));
  });
