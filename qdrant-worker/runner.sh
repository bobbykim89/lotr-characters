#!/bin/sh
set -e

# Runner script that:
# 1) Installs requirements.txt from /scripts if present
# 2) Waits for Qdrant to be available
# 3) Ensures /dist exists (writable)
# 4) Runs scrape_data.py then setup_qdrant.py (sequential)

# Environment variables are read from docker.env via docker-compose env_file
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

echo "Runner starting..."
echo "QDRANT_URL=${QDRANT_URL}"

# Install requirements if present in mounted src or repo root
if [ -f /scripts/requirements.txt ]; then
  REQ_PATH="/scripts/requirements.txt"
elif [ -f /requirements.txt ]; then
  REQ_PATH="/requirements.txt"
else
  REQ_PATH=""
fi

if [ -n "${REQ_PATH}" ]; then
  echo "Installing Python requirements from ${REQ_PATH}"
  pip install --upgrade pip setuptools wheel
  pip install -r "${REQ_PATH}"
else
  echo "No requirements file found. Skipping pip install."
fi

echo "Waiting for Qdrant at ${QDRANT_URL} ..."

TRIES=0
MAX_TRIES=180   # allow up to ~6 minutes (adjust if needed)
SLEEP=2

check_endpoint_python() {
  endpoint="$1"
  python - <<PY
import sys, os, urllib.request, urllib.error, json
endpoint = os.environ.get("QDRANT_URL")
api_key = os.environ.get("QDRANT_API_KEY", "") or None
req = urllib.request.Request(endpoint + '/collections')
if api_key:
    req.add_header("api-key", api_key)
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        code = resp.getcode()
        body = resp.read(512)
        print(f"PY_OK: status={code}")
        sys.exit(0)
except urllib.error.HTTPError as e:
    # HTTP status but non-2xx
    try:
        msg = e.read(512).decode('utf-8', errors='replace')
    except Exception:
        msg = '<no-body>'
    print(f"PY_HTTPERR: status={e.code} body={msg}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"PY_ERR: {str(e)}", file=sys.stderr)
    sys.exit(1)
PY
  return $?
}

until (
  export ENDPOINT="${QDRANT_URL%/}/collections"
  check_endpoint_python "${QDRANT_URL%/}/collections" 2>/dev/null
) || [ $TRIES -ge $MAX_TRIES ]; do
  TRIES=$((TRIES+1))
  echo "Qdrant not ready (try ${TRIES}/${MAX_TRIES}) — sleeping ${SLEEP}s..."
  sleep ${SLEEP}
done

if [ $TRIES -ge $MAX_TRIES ]; then
  echo "Timed out waiting for Qdrant at ${QDRANT_URL} after ${MAX_TRIES} attempts."
  echo "=== Diagnostics ==="
  echo "1) Inspect qdrant container logs on the host:"
  echo "   docker-compose logs qdrant --tail=200"
  echo "2) Check container status & ports:"
  echo "   docker ps --filter name=qdrant --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
  echo "3) Try contacting Qdrant from host (use API key if set):"
  if [ -n "${QDRANT_API_KEY}" ]; then
    echo "   curl -v -H \"api-key: ${QDRANT_API_KEY}\" http://localhost:6333/collections"
  else
    echo "   curl -v http://localhost:6333/collections"
  fi
  exit 1
fi

echo "Qdrant is responding."

# Ensure dist directory exists and is writable
if [ ! -d "${DIST_DIR}" ]; then
  echo "Creating ${DIST_DIR}"
  mkdir -p "${DIST_DIR}"
  chmod 755 "${DIST_DIR}"
fi

# Export convenience env vars for your scripts
export ASSETS_DIR DIST_DIR QDRANT_URL QDRANT_API_KEY OPENAI_API_KEY JINA_API_KEY

# Run setup_qdrant.py
if [ -f /scripts/setup_qdrant.py ]; then
  echo "Running /scripts/setup_qdrant.py..."
  python /scripts/setup_qdrant.py
else
  echo "Error: /scripts/setup_qdrant.py not found. Exiting."
  exit 3
fi


echo "Runner completed successfully. Processed outputs should be in ${DIST_DIR} (if your scripts write there)."