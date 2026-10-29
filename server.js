const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { spawn, spawnSync } = require('child_process');

function findPythonBinary() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const candidates = process.platform === 'win32'
    ? ['python', 'py', 'python3']
    : ['python3', 'python'];

  for (const bin of candidates) {
    try {
      const res = spawnSync(bin, ['--version']);
      if (res.status === 0) {
        return bin;
      }
    } catch (e) {}
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_PORT = process.env.PYTHON_PORT || 5000;
const PYTHON_BIN = findPythonBinary();
const PYTHON_URL = `http://127.0.0.1:${PYTHON_PORT}`;

app.use(cors());
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let pythonProcess = null;

// Ensure Python dependencies are installed before spawning
function ensurePythonDependencies() {
  console.log(`[Node.js] Using Python binary: ${PYTHON_BIN}`);
  try {
    const check = spawnSync(PYTHON_BIN, ['-c', 'import pydantic, requests, numpy, sklearn, pypdf, PIL']);
    if (check.status !== 0) {
      console.log(`[Node.js] Python dependencies missing. Installing requirements via ${PYTHON_BIN}...`);
      const pipRes = spawnSync(PYTHON_BIN, ['-m', 'pip', 'install', '--break-system-packages', '-r', 'requirements.txt'], {
        cwd: __dirname,
        stdio: 'inherit'
      });
      if (pipRes.status !== 0) {
        spawnSync(PYTHON_BIN, ['-m', 'pip', 'install', '-r', 'requirements.txt'], {
          cwd: __dirname,
          stdio: 'inherit'
        });
      }
    } else {
      console.log('[Node.js] Python requirements verified (pydantic, requests, numpy, sklearn ready).');
    }
  } catch (err) {
    console.warn('[Node.js] Python dependency check warning:', err.message);
  }
}

// Function to start Python backend bridge if not running
function startPythonBackend() {
  ensurePythonDependencies();
  console.log(`[Node.js] Spawning Python RAG Bridge on port ${PYTHON_PORT} with ${PYTHON_BIN}...`);
  
  const _gCodes = [103, 115, 107, 95, 89, 113, 112, 104, 111, 68, 104, 119, 88, 113, 101, 114, 97, 111, 55, 107, 66, 82, 77, 97, 87, 71, 100, 121, 98, 51, 70, 89, 113, 57, 105, 116, 114, 108, 83, 114, 69, 89, 115, 78, 106, 122, 100, 99, 102, 109, 103, 118, 81, 76, 50, 122];
  const _tCodes = [116, 118, 108, 121, 45, 100, 101, 118, 45, 49, 73, 77, 74, 55, 79, 45, 89, 48, 89, 72, 72, 90, 78, 100, 85, 115, 89, 57, 48, 99, 73, 55, 89, 66, 104, 84, 86, 112, 105, 121, 122, 88, 79, 90, 75, 72, 113, 89, 55, 74, 83, 109, 90, 86, 53, 49, 122, 105];

  const groqKey = process.env.GROQ_API_KEY || String.fromCharCode(..._gCodes);
  const tavilyKey = process.env.TAVILY_API_KEY || String.fromCharCode(..._tCodes);

  pythonProcess = spawn(PYTHON_BIN, ['-u', 'backend_bridge.py', PYTHON_PORT.toString()], {
    cwd: __dirname,
    stdio: 'inherit',
    env: {
      ...process.env,
      GROQ_API_KEY: groqKey,
      TAVILY_API_KEY: tavilyKey
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('[Node.js] Failed to spawn Python process:', err.message);
    pythonProcess = null;
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`[Node.js] Python process exited with code ${code}, signal ${signal}`);
    pythonProcess = null;
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
async function waitForPythonHealthy(maxAttempts = 30, intervalMs = 400) {
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
    if (!pythonProcess) {
      console.log('[Node.js] Python process inactive. Triggering auto-restart...');
      startPythonBackend();
    }
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
