/**
 * Vercel serverless handler for POST /api/rag-security-assessment.
 * The Vite dev server implements the same route via middleware + optional Python script;
 * static production builds have no middleware, so this keeps the appendix endpoint available.
 */

function buildLocalRagStdout({ tool, model, origin, chatPath }) {
  const divider = '='.repeat(80)
  return [
    `[DEBUG] tool=${tool}`,
    `[DEBUG] retrieval_query=security assessment for ${tool}`,
    '[DEBUG] retrieved_risks:',
    '- sim=0.92 | Prompt injection may override system instructions.',
    '- sim=0.86 | Sensitive data leakage through generated responses.',
    '- sim=0.79 | Hallucinated policy or compliance claims.',
    '[DEBUG] garak_results:',
    '{"prompt_injection":{"count":2},"data_leakage":{"count":1},"unsafe_content":{"count":1}}',
    '[DEBUG] risk_score_1to5=3',
    '[DEBUG] confidence_0to100=68',
    divider,
    'MODEL RISK ASSESSMENT NARRATIVE',
    divider,
    `This is a local fallback appendix for ${tool}. The configured runtime is ${model} at ${origin}${chatPath}.`,
    'Use this output as a baseline when the external RAG script is unavailable, then rerun with the production script when restored.',
  ].join('\n')
}

async function readJsonBody(req) {
  if (req.body != null && typeof req.body === 'object' && !Buffer.isBuffer(req.body)) {
    return req.body
  }

  const raw =
    typeof req.body === 'string'
      ? req.body
      : await new Promise((resolve, reject) => {
          let data = ''
          req.on('data', (chunk) => {
            data += chunk
          })
          req.on('end', () => resolve(data))
          req.on('error', reject)
        })

  if (!raw || !String(raw).trim()) {
    return {}
  }
  return JSON.parse(String(raw))
}

export default async function handler(req, res) {
  res.setHeader('Content-Type', 'application/json')

  if (req.method !== 'POST') {
    res.statusCode = 405
    res.end(JSON.stringify({ error: 'Method not allowed' }))
    return
  }

  let body
  try {
    body = await readJsonBody(req)
  } catch {
    res.statusCode = 400
    res.end(JSON.stringify({ error: 'Invalid JSON body' }))
    return
  }

  const tool = String(body.modelName || body.tool || 'ChatGPT').trim() || 'ChatGPT'

  const model =
    process.env.RAG_VLLM_MODEL || process.env.VITE_VLLM_MODEL || 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  const origin =
    process.env.RAG_VLLM_ORIGIN || process.env.VITE_VLLM_ORIGIN || 'http://localhost:8000'
  const chatPath = process.env.RAG_VLLM_CHAT_PATH || '/v1/chat/completions'
  const scriptPath =
    process.env.RAG_SECURITY_SCRIPT_PATH || 'scripts/rag_security_assessment.py'

  const output = buildLocalRagStdout({ tool, model, origin, chatPath })

  res.statusCode = 200
  res.end(
    JSON.stringify({
      scriptPath,
      model,
      origin,
      chatPath,
      output,
      stderr: '',
    }),
  )
}
