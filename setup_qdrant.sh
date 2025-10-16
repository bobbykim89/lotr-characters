#!/bin/sh
set -eu

# ------------------------
# setup_qdrant.sh
# - Stops & removes existing containers for this compose project
# - Starts qdrant
# - Waits for qdrant to be ready
# - Runs qdrant-worker (one-off) to execute local scripts
# ------------------------

# Choose docker-compose command (support docker-compose or docker compose)
if command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
elif docker compose version >/dev/null 2>&1; then
  DC="docker compose"
else
  echo "ERROR: neither 'docker-compose' nor 'docker compose' found in PATH."
  exit 1
fi

echo "Using: $DC"

# Load environment variables from docker.env if present, exporting them
if [ -f ./docker.env ]; then
  echo "Loading variables from ./docker.env"
  # export everything in docker.env
  # shellcheck disable=SC1091
  set -a
  . ./docker.env
  set +a
else
  echo "No docker.env file found in project root — continuing without sourcing it."
fi

# Ensure we run from repo root where docker-compose.yml exists
if [ ! -f docker-compose.yml ] && [ ! -f docker-compose.yaml ]; then
  echo "ERROR: docker-compose.yml not found in current directory. Run this script from the repo root."
  exit 2
fi

echo "Stopping and removing compose services (if any)..."
# Stop and remove containers, networks, volumes created by compose in this folder
$DC down -v --remove-orphans || true

# Additionally remove any stray containers whose name contains "qdrant" to avoid conflicts
echo "Removing stray containers containing 'qdrant' in their name (if any)..."
for cid in $(docker ps -a --filter "name=qdrant" --format '{{.ID}}'); do
  echo " -> removing container id $cid"
  docker rm -f "$cid" || true
done

# Start qdrant service
echo "Starting Qdrant (detached)..."
$DC up -d qdrant

# Optional: try to build qdrant-worker if it's buildable (no-op if not defined)
echo "Attempting to build qdrant-worker (if a build is defined)..."
# ignore build errors so we don't fail on simple image usage
$DC build qdrant-worker || true

# Wait for Qdrant to respond on localhost:6333/collections
QDRANT_HOST="${QDRANT_URL:-http://localhost:6333}"
# If docker.env provided QDRANT_URL that points to qdrant service name (http://qdrant:6333),
# we still want to test from host -> use localhost if host mapping exists.
# We'll attempt localhost first, if it fails and QDRANT_HOST is set to qdrant name, try that only from inside containers.
TEST_URL="http://localhost:6333/collections"

# If user supplied QDRANT_URL in docker.env and it's not localhost, keep fallback to it for messaging
echo "Waiting for Qdrant to be available at ${TEST_URL} (host) ..."

TRIES=0
MAX_TRIES=60
SLEEP=2

# Prepare curl auth header if QDRANT_API_KEY is set
if [ -n "${QDRANT_API_KEY:-}" ]; then
  AUTH_HEADER="-H Authorization: Bearer ${QDRANT_API_KEY}"
else
  AUTH_HEADER=""
fi

until sh -c "curl -s $AUTH_HEADER ${TEST_URL} >/dev/null 2>&1" || [ $TRIES -ge $MAX_TRIES ]; do
  TRIES=$((TRIES+1))
  echo "  qdrant not ready yet (attempt $TRIES/$MAX_TRIES) — sleeping ${SLEEP}s"
  sleep $SLEEP
done

if [ $TRIES -ge $MAX_TRIES ]; then
  echo "ERROR: timed out waiting for Qdrant at ${TEST_URL}."
  echo "Check 'docker ps' and container logs: ${DC} ps; ${DC} logs qdrant"
  exit 3
fi

echo "Qdrant appears ready on host port 6333."

# Run the qdrant-worker one-off to process and upsert data
echo "Running qdrant-worker (one-off). This will execute your scripts and then exit..."
# Using --rm keeps things tidy. This will stream logs to your terminal.
$DC run --rm qdrant-worker

echo "qdrant-worker finished."

echo "Done. Qdrant is running (docker-compose up -d qdrant). If you want to stop everything, run: ${DC} down -v"