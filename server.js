const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_PORT = process.env.PYTHON_PORT || 5000;
const PYTHON_URL = `http://127.0.0.1:${PYTHON_PORT}`;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let pythonProcess = null;

// Function to start Python backend bridge if not running
function startPythonBackend() {
  console.log(`[Node.js] Spawning Python RAG Bridge on port ${PYTHON_PORT}...`);
  pythonProcess = spawn('python', ['-u', 'backend_bridge.py', PYTHON_PORT.toString()], {
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

// Check if Python bridge is healthy, else start it
async function ensurePythonRunning() {
  return new Promise((resolve) => {
    const req = http.get(`${PYTHON_URL}/api/health`, (res) => {
      if (res.statusCode === 200) {
        console.log('[Node.js] Python backend is already running and healthy.');
        resolve(true);
      } else {
        startPythonBackend();
        resolve(false);
      }
    });

    req.on('error', () => {
      startPythonBackend();
      resolve(false);
    });

    req.setTimeout(1500, () => {
      req.destroy();
      startPythonBackend();
      resolve(false);
    });
  });
}

// Proxy helper
function proxyToPython(req, res, targetPath, method = 'GET', body = null) {
  const options = {
    hostname: '127.0.0.1',
    port: PYTHON_PORT,
    path: targetPath,
    method: method,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
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

  if (body) {
    pyReq.write(JSON.stringify(body));
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
  app.listen(PORT, () => {
    console.log(`\n=======================================================`);
    console.log(`🚀 SELF-INSTRUCT Interactive AI Assistant running!`);
    console.log(`👉 OPEN THIS IN YOUR BROWSER: http://localhost:${PORT}`);
    console.log(`👉 OR: http://127.0.0.1:${PORT}`);
    console.log(`⚠️  NOTE: Port ${PYTHON_PORT} is the internal Python API bridge.`);
    console.log(`   Always use port ${PORT} for the interactive UI.`);
    console.log(`=======================================================\n`);
  });
});
