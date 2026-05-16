import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn } from 'node:child_process'

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for']
  if (typeof forwarded === 'string' && forwarded.length > 0) {
    return forwarded.split(',')[0].trim()
  }
  if (Array.isArray(forwarded) && forwarded[0]) {
    return String(forwarded[0]).trim()
  }
  return req.socket?.remoteAddress || 'unknown'
}

/**
 * Simple per-IP fixed window limiter (in-memory; one process / one dev server).
 */
function createFixedWindowLimiter(max, windowMs) {
  const buckets = new Map()
  return function checkLimit(ip) {
    const now = Date.now()
    let b = buckets.get(ip)
    if (!b || now >= b.resetAt) {
      buckets.set(ip, { count: 1, resetAt: now + windowMs })
      return { allowed: true }
    }
    b.count += 1
    if (b.count > max) {
      return {
        allowed: false,
        retryAfterSeconds: Math.max(1, Math.ceil((b.resetAt - now) / 1000)),
      }
    }
    return { allowed: true }
  }
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
    })
    req.on('end', () => {
      if (!body) {
        resolve({})
        return
      }
      try {
        resolve(JSON.parse(body))
      } catch (error) {
        reject(new Error(`Invalid JSON body: ${error.message}`))
      }
    })
    req.on('error', reject)
  })
}

function runScript({
  scriptPath,
  tool,
  model,
  origin,
  chatPath,
  timeoutMs = 120000,
}) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      'python3',
      [scriptPath],
      {
        env: {
          ...process.env,
          TOOL_UNDER_TEST: tool,
          VLLM_MODEL: model,
          VLLM_ORIGIN: origin,
          VLLM_CHAT_PATH: chatPath,
        },
      },
    )

    let stdout = ''
    let stderr = ''
    let finished = false

    const timeout = setTimeout(() => {
      if (!finished) {
        finished = true
        child.kill('SIGTERM')
        reject(new Error(`Script timed out after ${timeoutMs}ms`))
      }
    }, timeoutMs)

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })
    child.on('error', (error) => {
      if (finished) return
      finished = true
      clearTimeout(timeout)
      reject(error)
    })
    child.on('close', (code) => {
      if (finished) return
      finished = true
      clearTimeout(timeout)
      if (code !== 0) {
        reject(
          new Error(
            `Script exited with code ${code}. ${stderr || 'No stderr output.'}`,
          ),
        )
        return
      }
      resolve({ stdout: stdout.trim(), stderr: stderr.trim() })
    })
  })
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const vllmOrigin = env.VITE_VLLM_ORIGIN || 'http://localhost:8000'
  const regularLlmModel = env.REGULAR_LLM_MODEL || 'gpt-4o-mini'
  const regularLlmApiKey = env.REGULAR_LLM_API_KEY || ''
  const regularLlmOrigin =
    env.REGULAR_LLM_ORIGIN || 'https://api.openai.com'
  const regularLlmChatPath =
    env.REGULAR_LLM_CHAT_PATH || '/v1/chat/completions'
  const ragScriptPath =
    env.RAG_SECURITY_SCRIPT_PATH ||
    '/Users/joshilano/Downloads/rag_security_assessment.py'
  const ragVllmModel =
    env.RAG_VLLM_MODEL ||
    env.VITE_VLLM_MODEL ||
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  const ragVllmOrigin = env.RAG_VLLM_ORIGIN || env.VITE_VLLM_ORIGIN || vllmOrigin
  const ragVllmChatPath = env.RAG_VLLM_CHAT_PATH || '/v1/chat/completions'

  const regularLlmRateLimitMax = Number(env.REGULAR_LLM_RATE_LIMIT_MAX) || 30
  const regularLlmRateWindowMs =
    Number(env.REGULAR_LLM_RATE_LIMIT_WINDOW_MS) || 60_000
  const ragRateLimitMax = Number(env.RAG_RATE_LIMIT_MAX) || 5
  const ragRateWindowMs = Number(env.RAG_RATE_LIMIT_WINDOW_MS) || 60_000

  function attachRateLimitedApiRoutes(server) {
    const checkRegularLlmLimit = createFixedWindowLimiter(
      regularLlmRateLimitMax,
      regularLlmRateWindowMs,
    )
    const checkRagLimit = createFixedWindowLimiter(
      ragRateLimitMax,
      ragRateWindowMs,
    )

    server.middlewares.use('/api/regular-chat', async (req, res) => {
      if (req.method !== 'POST') {
        res.statusCode = 405
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ error: 'Method not allowed' }))
        return
      }

      const ip = getClientIp(req)
      const regularLimit = checkRegularLlmLimit(ip)
      if (!regularLimit.allowed) {
        res.statusCode = 429
        res.setHeader('Content-Type', 'application/json')
        res.setHeader(
          'Retry-After',
          String(regularLimit.retryAfterSeconds ?? 60),
        )
        res.end(
          JSON.stringify({
            error:
              'Too many regular LLM requests. Please wait and try again.',
          }),
        )
        return
      }

      if (!regularLlmApiKey) {
        res.statusCode = 503
        res.setHeader('Content-Type', 'application/json')
        res.end(
          JSON.stringify({
            error:
              'Regular LLM fallback is not configured. Set REGULAR_LLM_API_KEY.',
          }),
        )
        return
      }

      try {
        const body = await readRequestBody(req)
        const payload = {
          ...body,
          model: body.model || regularLlmModel,
        }
        const response = await fetch(
          `${regularLlmOrigin}${regularLlmChatPath}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${regularLlmApiKey}`,
            },
            body: JSON.stringify(payload),
          },
        )
        const text = await response.text()

        res.statusCode = response.status
        res.setHeader('Content-Type', 'application/json')
        res.end(text)
      } catch (error) {
        res.statusCode = 500
        res.setHeader('Content-Type', 'application/json')
        res.end(
          JSON.stringify({
            error:
              error instanceof Error
                ? error.message
                : 'Unknown regular LLM fallback error',
          }),
        )
      }
    })

    server.middlewares.use('/api/rag-security-assessment', async (req, res) => {
      if (req.method !== 'POST') {
        res.statusCode = 405
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ error: 'Method not allowed' }))
        return
      }

      const ip = getClientIp(req)
      const ragLimit = checkRagLimit(ip)
      if (!ragLimit.allowed) {
        res.statusCode = 429
        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Retry-After', String(ragLimit.retryAfterSeconds ?? 60))
        res.end(
          JSON.stringify({
            error:
              'Too many RAG security assessment requests. Please wait and try again.',
          }),
        )
        return
      }

      try {
        const body = await readRequestBody(req)
        const tool = String(body.modelName || body.tool || 'ChatGPT').trim()

        const result = await runScript({
          scriptPath: ragScriptPath,
          tool: tool || 'ChatGPT',
          model: ragVllmModel,
          origin: ragVllmOrigin,
          chatPath: ragVllmChatPath,
        })

        res.statusCode = 200
        res.setHeader('Content-Type', 'application/json')
        res.end(
          JSON.stringify({
            scriptPath: ragScriptPath,
            model: ragVllmModel,
            origin: ragVllmOrigin,
            chatPath: ragVllmChatPath,
            output: result.stdout,
            stderr: result.stderr,
          }),
        )
      } catch (error) {
        res.statusCode = 500
        res.setHeader('Content-Type', 'application/json')
        res.end(
          JSON.stringify({
            error:
              error instanceof Error
                ? error.message
                : 'Unknown script execution error',
          }),
        )
      }
    })
  }

  return {
    plugins: [
      react(),
      {
        name: 'rag-security-assessment-api',
        configureServer(server) {
          attachRateLimitedApiRoutes(server)
        },
        configurePreviewServer(server) {
          attachRateLimitedApiRoutes(server)
        },
      },
    ],
    server: {
      proxy: {
        '/api/chat': {
          target: vllmOrigin,
          changeOrigin: true,
          rewrite: () => '/v1/chat/completions',
        },
      },
    },
  }
})
