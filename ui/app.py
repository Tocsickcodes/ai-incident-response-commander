import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify
from main import run_simulation

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Incident Response Commander</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@400;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --red:    #ff2d2d;
    --orange: #ff8c00;
    --green:  #00ff88;
    --blue:   #00bfff;
    --yellow: #ffe600;
    --bg:     #0a0c10;
    --panel:  #10141c;
    --border: #1e2535;
    --text:   #c9d1e0;
    --dim:    #4a5568;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Share Tech Mono', monospace;
    min-height: 100vh;
    padding: 2rem;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 999;
  }

  header { text-align: center; margin-bottom: 2.5rem; }

  h1 {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: clamp(1.6rem, 4vw, 2.8rem);
    color: var(--red);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-shadow: 0 0 30px rgba(255,45,45,0.5);
  }

  .subtitle {
    color: var(--dim);
    font-size: 0.85rem;
    margin-top: 0.4rem;
    letter-spacing: 0.15em;
  }

  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 1.5rem;
    max-width: 1100px;
    margin: 0 auto;
  }

  .sidebar { display: flex; flex-direction: column; gap: 1rem; }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem;
  }

  .panel-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
  }

  .scenario-tag {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--border);
  }
  .scenario-tag:last-child { border-bottom: none; }
  .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .dot-red    { background: var(--red);    box-shadow: 0 0 6px var(--red); }
  .dot-orange { background: var(--orange); box-shadow: 0 0 6px var(--orange); }
  .dot-blue   { background: var(--blue);   box-shadow: 0 0 6px var(--blue); }

  .action-list { list-style: none; }
  .action-list li { font-size: 0.8rem; padding: 0.3rem 0; color: var(--dim); }
  .action-list li::before { content: '▸ '; color: var(--orange); }

  .scoring-row { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 0.25rem 0; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }

  #run-btn {
    width: 100%;
    padding: 1rem;
    background: transparent;
    border: 2px solid var(--red);
    color: var(--red);
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }
  #run-btn:hover:not(:disabled) {
    background: var(--red);
    color: #000;
    box-shadow: 0 0 20px rgba(255,45,45,0.4);
  }
  #run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .main-col { display: flex; flex-direction: column; gap: 1rem; }

  #stats-bar {
    display: none;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }

  .stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .stat-label { font-size: 0.65rem; color: var(--dim); letter-spacing: 0.15em; text-transform: uppercase; }
  .stat-value { font-family: 'Exo 2', sans-serif; font-size: 1.6rem; font-weight: 900; margin-top: 0.3rem; }
  .stat-green  { color: var(--green);  text-shadow: 0 0 15px rgba(0,255,136,0.4); }
  .stat-blue   { color: var(--blue);   text-shadow: 0 0 15px rgba(0,191,255,0.4); }
  .stat-yellow { color: var(--yellow); text-shadow: 0 0 15px rgba(255,230,0,0.4); }

  #output {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem;
    min-height: 420px;
    font-size: 0.82rem;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-y: auto;
    flex: 1;
  }

  .placeholder { color: var(--dim); text-align: center; margin-top: 4rem; line-height: 2; }

  .step-block   { margin-bottom: 1.2rem; }
  .step-header  { font-family: 'Exo 2', sans-serif; font-weight: 700; color: var(--orange);
                  font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.4rem; }
  .step-action  { color: var(--blue); }
  .step-msg     { color: var(--text); }
  .reward-pos   { color: var(--green); }
  .reward-neg   { color: var(--red); }

  .log-block, .metric-block {
    margin-top: 0.6rem;
    padding: 0.7rem;
    background: rgba(255,255,255,0.03);
    border-left: 2px solid var(--border);
    font-size: 0.77rem;
    color: var(--dim);
  }
  .lbl { color: var(--yellow); font-size: 0.7rem; letter-spacing: 0.1em; margin-bottom: 0.3rem; }

  hr.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

  .final-score {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 1.1rem;
    text-align: center;
    padding: 1rem;
    border-radius: 4px;
    margin-top: 0.5rem;
  }
  .grade-excellent { background: rgba(0,255,136,0.08); color: var(--green); border: 1px solid var(--green); }
  .grade-good      { background: rgba(0,191,255,0.08); color: var(--blue);  border: 1px solid var(--blue); }
  .grade-ok        { background: rgba(255,230,0,0.08); color: var(--yellow);border: 1px solid var(--yellow); }
  .grade-poor      { background: rgba(255,45,45,0.08); color: var(--red);   border: 1px solid var(--red); }

  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid var(--dim);
    border-top-color: var(--red);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 700px) {
    .layout { grid-template-columns: 1fr; }
    #stats-bar { grid-template-columns: repeat(3, 1fr); }
  }
</style>
</head>
<body>

<header>
  <h1>🚨 AI Incident Response Commander</h1>
  <p class="subtitle">AUTONOMOUS AGENT &nbsp;·&nbsp; REAL-TIME INVESTIGATION &nbsp;·&nbsp; ROOT CAUSE ANALYSIS</p>
</header>

<div class="layout">

  <aside class="sidebar">
    <div class="panel">
      <div class="panel-title">Scenarios</div>
      <div class="scenario-tag"><span class="dot dot-red"></span>Memory Leak</div>
      <div class="scenario-tag"><span class="dot dot-orange"></span>Bad Deployment</div>
      <div class="scenario-tag"><span class="dot dot-blue"></span>Network Issue</div>
    </div>

    <div class="panel">
      <div class="panel-title">Agent Actions</div>
      <ul class="action-list">
        <li>inspect_logs</li>
        <li>check_metrics</li>
        <li>restart_service</li>
        <li>rollback_deployment</li>
        <li>escalate</li>
      </ul>
    </div>

    <div class="panel">
      <div class="panel-title">Scoring</div>
      <div class="scoring-row"><span>Correct fix</span><span class="pos">+100</span></div>
      <div class="scoring-row"><span>Root cause ID</span><span class="pos">+50</span></div>
      <div class="scoring-row"><span>Per step</span><span class="neg">-2</span></div>
      <div class="scoring-row"><span>Unnecessary action</span><span class="neg">-10</span></div>
      <div class="scoring-row"><span>Wrong fix</span><span class="neg">-30</span></div>
    </div>

    <button id="run-btn" onclick="runSim()">&#9654; Run Simulation</button>
  </aside>

  <main class="main-col">
    <div id="stats-bar">
      <div class="stat-card">
        <div class="stat-label">Scenario</div>
        <div class="stat-value stat-blue" id="stat-scenario" style="font-size:1rem;margin-top:0.5rem">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Steps Taken</div>
        <div class="stat-value stat-yellow" id="stat-steps">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Final Score</div>
        <div class="stat-value" id="stat-score">—</div>
      </div>
    </div>

    <div id="output">
      <div class="placeholder">
        Click <strong>Run Simulation</strong> to deploy the AI agent<br>
        into a live production incident scenario.
      </div>
    </div>
  </main>
</div>

<script>
async function runSim() {
  const btn = document.getElementById('run-btn');
  const out = document.getElementById('output');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Investigating...';
  out.innerHTML = '<span style="color:var(--dim)">Agent dispatched. Awaiting results...</span>';
  document.getElementById('stats-bar').style.display = 'none';

  try {
    const res  = await fetch('/run');
    const data = await res.json();
    renderResult(data);
  } catch(e) {
    out.innerHTML = '<span style="color:var(--red)">Error contacting server: ' + e.message + '</span>';
  }

  btn.disabled = false;
  btn.innerHTML = '&#9654; Run Simulation';
}

function renderResult(data) {
  const out = document.getElementById('output');

  // Stats bar
  document.getElementById('stat-scenario').textContent = data.scenario;
  document.getElementById('stat-steps').textContent    = data.total_steps;
  const scoreEl = document.getElementById('stat-score');
  scoreEl.textContent  = (data.total_score >= 0 ? '+' : '') + data.total_score;
  scoreEl.className    = 'stat-value ' + (data.total_score >= 100 ? 'stat-green' : data.total_score >= 0 ? 'stat-yellow' : 'neg');
  document.getElementById('stats-bar').style.display = 'grid';

  let html = '';
  html += '<div style="color:var(--dim);font-size:0.7rem;letter-spacing:0.15em;margin-bottom:1rem">';
  html += 'INCIDENT DETECTED &nbsp;&middot;&nbsp; SCENARIO: ';
  html += '<span style="color:var(--blue)">' + esc(data.scenario.toUpperCase()) + '</span></div>';

  for (const step of data.steps) {
    const rc  = step.reward >= 0 ? 'reward-pos' : 'reward-neg';
    const rs  = (step.reward >= 0 ? '+' : '') + step.reward;

    html += '<div class="step-block">';
    html += '<div class="step-header">&#9472;&#9472; Step ' + step.step + ' ' + '&#9472;'.repeat(34) + '</div>';
    html += '<div class="step-action">&#x1F916; ACTION: ' + esc(step.action.replace(/_/g,' ').toUpperCase()) + '</div>';
    html += '<div class="step-msg">' + esc(step.message) + '</div>';
    html += '<div class="' + rc + '">&#x1F4B0; Reward: ' + rs + ' pts</div>';

    if (step.logs) {
      html += '<div class="log-block"><div class="lbl">&#x1F4CB; LOGS REVEALED</div>' + esc(step.logs) + '</div>';
    }
    if (step.metrics) {
      let m = '';
      for (const [k, v] of Object.entries(step.metrics)) {
        m += k.padEnd(30) + v + '\\n';
      }
      html += '<div class="metric-block"><div class="lbl">&#x1F4CA; METRICS REVEALED</div>' + esc(m) + '</div>';
    }
    html += '</div>';
  }

  const score = data.total_score;
  let gc, gt;
  if      (score >= 100) { gc = 'grade-excellent'; gt = '&#x1F3C6; EXCELLENT &mdash; Perfect incident response'; }
  else if (score >= 50)  { gc = 'grade-good';      gt = '&#x2705; GOOD &mdash; Solid investigation'; }
  else if (score >= 0)   { gc = 'grade-ok';        gt = '&#x26A0;&#xFE0F; ACCEPTABLE &mdash; Room for improvement'; }
  else                   { gc = 'grade-poor';       gt = '&#x274C; POOR &mdash; Incident mishandled'; }

  html += '<hr class="divider">';
  html += '<div class="final-score ' + gc + '">' + gt + '<br>';
  html += '<span style="font-size:0.8rem;font-weight:400;font-family:monospace">';
  html += 'Steps: ' + data.total_steps + ' &nbsp;&middot;&nbsp; Score: ' + (score >= 0 ? '+' : '') + score + ' pts</span></div>';

  out.innerHTML = html;
  out.scrollTop = 0;
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/run")
def run():
    """Run one simulation episode and return the result as JSON."""
    result = run_simulation(verbose=False)
    return jsonify(result)


@app.route("/reset", methods=["POST"])
def reset():
    """Required by OpenEnv validator — POST /reset returns 200 with fresh env state."""
    from env.environment import Environment
    env = Environment()
    obs = env.reset()
    return jsonify({
        "status":   "ok",
        "scenario": env.get_scenario_name(),
        "alerts":   obs.alerts,
        "message":  obs.message,
    }), 200


@app.route("/state", methods=["GET"])
def state():
    """GET /state — returns episode metadata (OpenEnv state() method via HTTP)."""
    from env.environment import Environment
    env = Environment()
    env.reset()
    s = env.state()
    return jsonify(s.to_dict()), 200


@app.route("/health")
def health():
    """Liveness check used by Docker HEALTHCHECK and pre-validation script."""
    return jsonify({"status": "ok", "service": "ai-incident-response-commander"}), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n\U0001F6A8  AI Incident Response Commander")
    print(f"    Open http://localhost:{port} in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=port)