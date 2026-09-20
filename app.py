"""Interactive Live Web Application & REST API for Support Ticket Classification & Prioritization.

Runs a zero-dependency local web server using Python's standard library (http.server).
Features:
- Sleek modern UI (Glassmorphic dark mode, responsive, animated probability meters)
- Real-time classification with calibrated confidence scores
- Uncertainty gating badge (<70% routes to human triage)
- Quick-fill sample tickets
- Embedded REST API: POST /api/predict
"""

import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys

# Ensure project modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.predict import TicketClassifier

# Initialize inference engine once at startup
classifier = TicketClassifier()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Support Ticket AI Triage | Live ML Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #111827;
      --card-bg: rgba(17, 24, 39, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-cyan: #00f2fe;
      --accent-blue: #4facfe;
      --critical: #ff3366;
      --high: #ff9900;
      --medium: #fbc02d;
      --low: #00e676;
      --bug: #e040fb;
      --billing: #00e5ff;
      --account: #ffab00;
      --feature: #00e676;
      --it: #2979ff;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at top right, #1a233a 0%, #0a0e17 50%, #05070d 100%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .container {
      max-width: 1050px;
      width: 100%;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(79, 172, 254, 0.12);
      border: 1px solid rgba(79, 172, 254, 0.3);
      padding: 0.35rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--accent-cyan);
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .badge-dot {
      width: 8px;
      height: 8px;
      background: var(--accent-cyan);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-cyan);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    h1 {
      font-size: 2.75rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 30%, #9ca3af 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.75rem;
    }

    .subtitle {
      font-size: 1.1rem;
      color: var(--text-muted);
      max-width: 650px;
      margin: 0 auto;
      line-height: 1.6;
    }

    /* Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1.75rem;
    }

    @media (max-width: 860px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 1.75rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .card-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    /* Textarea & Controls */
    textarea {
      width: 100%;
      height: 140px;
      background: rgba(10, 14, 23, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      padding: 1rem;
      color: #fff;
      font-family: inherit;
      font-size: 0.98rem;
      resize: vertical;
      transition: all 0.2s ease;
      line-height: 1.5;
    }

    textarea:focus {
      outline: none;
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.15);
    }

    .sample-chips {
      margin: 1.2rem 0;
    }

    .sample-chips-label {
      font-size: 0.82rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.6rem;
      display: block;
      font-weight: 600;
    }

    .chips-container {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.09);
      padding: 0.45rem 0.8rem;
      border-radius: 10px;
      font-size: 0.82rem;
      cursor: pointer;
      color: #d1d5db;
      transition: all 0.2s ease;
      white-space: nowrap;
    }

    .chip:hover {
      background: rgba(79, 172, 254, 0.18);
      border-color: var(--accent-cyan);
      color: #fff;
      transform: translateY(-1px);
    }

    .btn-classify {
      width: 100%;
      background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
      color: #05070d;
      font-weight: 700;
      font-size: 1.05rem;
      padding: 0.95rem;
      border: none;
      border-radius: 14px;
      cursor: pointer;
      transition: all 0.25s ease;
      box-shadow: 0 8px 25px rgba(0, 242, 254, 0.35);
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.6rem;
    }

    .btn-classify:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 35px rgba(0, 242, 254, 0.5);
    }

    .btn-classify:active {
      transform: translateY(0);
    }

    /* Results Section */
    .result-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 320px;
      color: var(--text-muted);
      text-align: center;
    }

    .result-empty svg {
      width: 54px;
      height: 54px;
      stroke: rgba(255, 255, 255, 0.2);
      margin-bottom: 1rem;
    }

    .prediction-container {
      display: none;
    }

    .prediction-row {
      margin-bottom: 1.5rem;
    }

    .label-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }

    .label-title {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 600;
    }

    .prediction-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      border-radius: 12px;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      margin-bottom: 0.6rem;
    }

    /* Progress bar */
    .meter-container {
      background: rgba(255, 255, 255, 0.08);
      height: 9px;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 0.4rem;
    }

    .meter-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
      width: 0%;
    }

    .confidence-text {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: var(--text-muted);
      text-align: right;
    }

    /* Priority Colors */
    .pri-Critical { background: rgba(255, 51, 102, 0.15); color: #ff4071; border: 1px solid rgba(255, 51, 102, 0.4); }
    .pri-High { background: rgba(255, 153, 0, 0.15); color: #ffa012; border: 1px solid rgba(255, 153, 0, 0.4); }
    .pri-Medium { background: rgba(251, 192, 45, 0.15); color: #ffd54f; border: 1px solid rgba(251, 192, 45, 0.4); }
    .pri-Low { background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.4); }

    .fill-Critical { background: linear-gradient(90deg, #ff3366, #ff6b8b); }
    .fill-High { background: linear-gradient(90deg, #ff9900, #ffb84d); }
    .fill-Medium { background: linear-gradient(90deg, #fbc02d, #ffee58); }
    .fill-Low { background: linear-gradient(90deg, #00e676, #69f0ae); }

    /* Category Colors */
    .cat-badge {
      background: rgba(79, 172, 254, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(0, 242, 254, 0.35);
    }
    .fill-Category { background: linear-gradient(90deg, #00f2fe, #4facfe); }

    /* Human Triage Alert */
    .alert-triage {
      background: rgba(255, 153, 0, 0.1);
      border: 1px solid rgba(255, 153, 0, 0.35);
      border-radius: 12px;
      padding: 0.85rem;
      font-size: 0.85rem;
      color: #ffb74d;
      margin-top: 1rem;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    /* Footer */
    footer {
      margin-top: 3rem;
      text-align: center;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    footer a {
      color: var(--accent-cyan);
      text-decoration: none;
      font-weight: 600;
    }
    footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-pill">
        <span class="badge-dot"></span>
        Production Classical Machine Learning
      </div>
      <h1>Support Ticket AI Triage</h1>
      <p class="subtitle">
        Joint category routing & severity prioritization powered by TF-IDF, Naive Bayes, and Calibrated Logistic Regression.
      </p>
    </header>

    <div class="dashboard-grid">
      <!-- Input Card -->
      <div class="card">
        <h2 class="card-title">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          Inbound Support Ticket
        </h2>

        <textarea id="ticketInput" placeholder="Enter customer ticket description or paste error logs..."></textarea>

        <div class="sample-chips">
          <span class="sample-chips-label">Quick Test Scenarios:</span>
          <div class="chips-container">
            <button class="chip" onclick="setSample(0)">💥 Crash / Outage (Bug)</button>
            <button class="chip" onclick="setSample(1)">💳 Double Charge (Billing)</button>
            <button class="chip" onclick="setSample(2)">🔒 2FA Failure (Access)</button>
            <button class="chip" onclick="setSample(3)">🌐 VPN Certificate (IT)</button>
            <button class="chip" onclick="setSample(4)">✨ Dark Mode (Feature)</button>
          </div>
        </div>

        <button class="btn-classify" id="btnClassify" onclick="classifyTicket()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          Classify & Prioritize
        </button>
      </div>

      <!-- Results Card -->
      <div class="card">
        <h2 class="card-title">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          Triage Verdict
        </h2>

        <div class="result-empty" id="resultEmpty">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
          <p>Submit a support ticket on the left to see live ML predictions & confidence calibration.</p>
        </div>

        <div class="prediction-container" id="predictionResult">
          <!-- Category Row -->
          <div class="prediction-row">
            <div class="label-header">
              <span class="label-title">Predicted Category</span>
            </div>
            <div class="prediction-badge cat-badge" id="catBadge">
              <span id="catText">Bug / System Error</span>
            </div>
            <div class="meter-container">
              <div class="meter-fill fill-Category" id="catMeter"></div>
            </div>
            <div class="confidence-text" id="catConf">96.5% Confidence</div>
          </div>

          <!-- Priority Row -->
          <div class="prediction-row">
            <div class="label-header">
              <span class="label-title">Predicted Priority / SLA</span>
            </div>
            <div class="prediction-badge" id="priBadge">
              <span id="priText">Critical</span>
            </div>
            <div class="meter-container">
              <div class="meter-fill" id="priMeter"></div>
            </div>
            <div class="confidence-text" id="priConf">85.2% Confidence</div>
          </div>

          <!-- Low Confidence Warning -->
          <div class="alert-triage" id="triageAlert" style="display: none;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            <span><strong>Low Confidence (< 70%):</strong> Flagged for human review.</span>
          </div>
        </div>
      </div>
    </div>

    <footer>
      Support Ticket Classification & Prioritization &bull; Classical ML Portfolio Project &bull;
      <a href="https://github.com/manojkumarj-dev/Support-Ticket-Classification-Prioritization" target="_blank">View on GitHub</a>
    </footer>
  </div>

  <script>
    const SAMPLES = [
      "Application crashes with 500 Internal Server Error when generating the annual tax reconciliation report in Reporting Engine. This is causing immediate revenue loss!",
      "Our corporate credit card was billed twice for invoice #94821. Total overcharge is $2,450. Please issue an immediate refund.",
      "Two-Factor Authentication (2FA) SMS verification code is not being delivered to my mobile phone. Completely locked out of our admin account!",
      "Corporate VPN gateway is rejecting connections with 'TLS Handshake Failed: certificate expired'. Remote engineers cannot access servers.",
      "Feature Request: Please add native Dark Mode support across the entire web application dashboard for night-shift operators."
    ];

    function setSample(index) {
      document.getElementById('ticketInput').value = SAMPLES[index];
      classifyTicket();
    }

    async function classifyTicket() {
      const text = document.getElementById('ticketInput').value.trim();
      if (!text) {
        alert("Please enter ticket text or choose a quick sample.");
        return;
      }

      const btn = document.getElementById('btnClassify');
      btn.disabled = true;
      btn.innerHTML = `Analyzing...`;

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert("Prediction failed: " + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Classify & Prioritize`;
      }
    }

    function renderResults(data) {
      document.getElementById('resultEmpty').style.display = 'none';
      const container = document.getElementById('predictionResult');
      container.style.display = 'block';

      // Category
      document.getElementById('catText').innerText = data.category;
      const catConfPct = Math.round((data.category_confidence || 0) * 100);
      document.getElementById('catMeter').style.width = `${catConfPct}%`;
      document.getElementById('catConf').innerText = `${catConfPct}% Confidence (Calibrated)`;

      // Priority
      const priText = document.getElementById('priText');
      const priBadge = document.getElementById('priBadge');
      const priMeter = document.getElementById('priMeter');
      priText.innerText = data.priority;

      priBadge.className = `prediction-badge pri-${data.priority}`;
      priMeter.className = `meter-fill fill-${data.priority}`;
      const priConfPct = Math.round((data.priority_confidence || 0) * 100);
      priMeter.style.width = `${priConfPct}%`;
      document.getElementById('priConf').innerText = `${priConfPct}% Confidence (Calibrated)`;

      // Low confidence alert
      const triageAlert = document.getElementById('triageAlert');
      if (data.priority_confidence < 0.70 || data.category_confidence < 0.70) {
        triageAlert.style.display = 'flex';
      } else {
        triageAlert.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""

class LiveAppHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler serving the SPA frontend and JSON prediction API."""

    def do_GET(self):
        """Serves the interactive single-page application."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        """REST API endpoint for real-time ticket classification."""
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                text = payload.get("text", "")
                result = classifier.predict(text)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress noisy request logs."""
        pass


def run_server(port: int = 8501):
    server = HTTPServer(("0.0.0.0", port), LiveAppHandler)
    print("=" * 60)
    print(f"🚀 Support Ticket AI Triage Live Server running at:")
    print(f"   Local URL:    http://localhost:{port}")
    print(f"   Network URL:  http://127.0.0.1:{port}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
