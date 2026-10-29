const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_PORT = process.env.PYTHON_PORT || 5000;
const PYTHON_BIN = process.env.PYTHON_BIN || 'python';
const PYTHON_URL = `http://127.0.0.1:${PYTHON_PORT}`;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let pythonProcess = null;

// Function to start Python backend bridge if not running
function startPythonBackend() {
  console.log(`[Node.js] Spawning Python RAG Bridge on port ${PYTHON_PORT} with ${PYTHON_BIN}...`);
  pythonProcess = spawn(PYTHON_BIN, ['-u', 'backend_bridge.py', PYTHON_PORT.toString()], {
    cwd: __dirname,
    stdio: 'inherit'
  });

  pythonProcess.on('error', (err) => {
    console.error('[Node.js] Failed to spawn Python process:', err.message);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`[Node.js] Python process exited with code ${code}, signal ${signal}`);
  });
}

// Check health of Python bridge
function checkPythonHealth() {
  return new Promise((resolve) => {
    const req = http.get(`${PYTHON_URL}/api/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(800, () => {
      req.destroy();
      resolve(false);
    });
  });
}

// Wait until Python bridge is healthy
async function waitForPythonHealthy(maxAttempts = 20, intervalMs = 400) {
  for (let i = 0; i < maxAttempts; i++) {
    const healthy = await checkPythonHealth();
    if (healthy) {
      console.log('[Node.js] Python backend bridge is initialized and ready.');
      return true;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  console.warn('[Node.js] Python backend took longer than usual to start.');
  return false;
}

// Check if Python bridge is healthy, else start it and wait
async function ensurePythonRunning() {
  const isRunning = await checkPythonHealth();
  if (isRunning) {
    console.log('[Node.js] Python backend is already running.');
    return true;
  }
  startPythonBackend();
  return await waitForPythonHealthy();
}

// Proxy helper
function proxyToPython(req, res, targetPath, method = 'GET', body = null) {
  const payloadStr = body ? JSON.stringify(body) : null;
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };

  if (payloadStr) {
    headers['Content-Length'] = Buffer.byteLength(payloadStr, 'utf-8');
  }

  const options = {
    hostname: '127.0.0.1',
    port: PYTHON_PORT,
    path: targetPath,
    method: method,
    headers: headers
  };

  const pyReq = http.request(options, (pyRes) => {
    res.status(pyRes.statusCode);
    Object.keys(pyRes.headers).forEach((h) => {
      res.setHeader(h, pyRes.headers[h]);
    });
    pyRes.pipe(res);
  });

  pyReq.on('error', (err) => {
    res.status(503).json({
      error: 'Python backend bridge is unavailable or starting up.',
      details: err.message
    });
  });

  if (payloadStr) {
    pyReq.write(payloadStr);
  }
  pyReq.end();
}

// Proxy API routes
app.get('/api/health', (req, res) => {
  proxyToPython(req, res, '/api/health', 'GET');
});

app.get('/api/threads', (req, res) => {
  proxyToPython(req, res, '/api/threads', 'GET');
});

app.get('/api/history', (req, res) => {
  const queryParam = req.url.includes('?') ? req.url.substring(req.url.indexOf('?')) : '';
  proxyToPython(req, res, `/api/history${queryParam}`, 'GET');
});

app.post('/api/query', (req, res) => {
  proxyToPython(req, res, '/api/query', 'POST', req.body);
});

app.post('/api/clear', (req, res) => {
  proxyToPython(req, res, '/api/clear', 'POST', req.body);
});

// Fallback for single page application routing
app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Clean shutdown
process.on('SIGINT', () => {
  console.log('\n[Node.js] Gracefully shutting down...');
  if (pythonProcess) {
    pythonProcess.kill();
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  process.exit(0);
});

// Start Express Server
ensurePythonRunning().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n=======================================================`);
    console.log(`⚡ BAASHA AGENT - ReAct Intelligence Engine running!`);
    console.log(`👉 BINDING ON: http://0.0.0.0:${PORT}`);
    console.log(`👉 LOCAL URL: http://localhost:${PORT}`);
    console.log(`⚠️  NOTE: Port ${PYTHON_PORT} is the internal Python API bridge.`);
    console.log(`   Always use port ${PORT} for the Baasha Agent UI.`);
    console.log(`=======================================================\n`);
  });
});
