# Dockerfile — AI Incident Response Commander
#
# Build:  docker build -t ai-incident-agent .
# Run:    docker run -p 5000:5000 ai-incident-agent
#         docker run -p 5000:5000 -e HF_TOKEN=gsk_... ai-incident-agent
#
# Then open http://localhost:5000 in your browser.

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Keeps Python output flushed straight to stdout (important for [STEP] logging)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker cache is reused when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Health check — used by Docker and the pre-validation script ───────────────
# Waits 10s before first check, retries every 5s, 3 retries before unhealthy
HEALTHCHECK --interval=5s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Default command — start the Flask web server ─────────────────────────────
CMD ["python", "ui/app.py"]