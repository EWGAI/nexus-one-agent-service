#!/usr/bin/env bash
# End-to-end smoke test: mint token -> ingest -> search -> chat.
# Usage: BASE_URL=http://localhost:8000 scripts/smoke_test.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PYTHON="${PYTHON:-python}"
DOMAIN="${DOMAIN:-hr}"
FACT="Employees accrue 1.75 days of paid leave per completed month of service."

say() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31mFAIL:\033[0m %s\n' "$1" >&2; exit 1; }

say "waiting for ${BASE_URL}/health"
for _ in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "${BASE_URL}/health" >/dev/null || fail "service never became healthy"

say "minting a token"
TOKEN="$("${PYTHON}" scripts/mint_token.py --quiet --sub smoke@example.com \
  --scopes "kb:*:read kb:*:write kb:admin agent:chat")"
AUTH="Authorization: Bearer ${TOKEN}"

say "unauthenticated requests must be rejected"
code="$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/kb/${DOMAIN}/search?q=test")"
[ "${code}" = "401" ] || fail "expected 401 without a token, got ${code}"

say "ingesting text into '${DOMAIN}'"
curl -fsS -X POST "${BASE_URL}/kb/${DOMAIN}/ingest/text" \
  -H "${AUTH}" -H 'Content-Type: application/json' \
  -d "{\"text\": \"${FACT}\", \"source\": \"smoke-policy\", \"tags\": {\"module\": \"leave\"}}" \
  | "${PYTHON}" -c '
import json,sys
data = json.load(sys.stdin)["result"]
assert data["chunks_added"] + data["duplicates_skipped"] > 0, data
print("   chunks_added=%s duplicates_skipped=%s" % (data["chunks_added"], data["duplicates_skipped"]))
'

say "searching '${DOMAIN}'"
curl -fsS -G "${BASE_URL}/kb/${DOMAIN}/search" -H "${AUTH}" \
  --data-urlencode 'q=how much paid leave do employees accrue' --data-urlencode 'k=3' \
  | "${PYTHON}" -c '
import json,sys
hits = json.load(sys.stdin)["hits"]
assert hits, "search returned no hits"
assert any("leave" in h["content"].lower() for h in hits), hits
assert all(h["domain"] == "'"${DOMAIN}"'" for h in hits), hits
print("   top hit score=%.4f source=%s" % (hits[0]["score"], hits[0]["metadata"].get("source")))
'

say "cross-domain isolation (finance must not see the ${DOMAIN} chunk)"
curl -fsS -G "${BASE_URL}/kb/finance/search" -H "${AUTH}" \
  --data-urlencode 'q=how much paid leave do employees accrue' \
  | "${PYTHON}" -c '
import json,sys
hits = json.load(sys.stdin)["hits"]
assert not any("accrue 1.75" in h["content"] for h in hits), hits
print("   finance returned %d unrelated hit(s)" % len(hits))
'

say "chatting with the agent"
curl -fsS -X POST "${BASE_URL}/agent/chat" -H "${AUTH}" -H 'Content-Type: application/json' \
  -d "{\"message\": \"How much paid leave do employees accrue?\", \"domains\": [\"${DOMAIN}\"]}" \
  | "${PYTHON}" -c '
import json,sys
data = json.load(sys.stdin)
assert data["answer"].strip(), data
assert data["sources"], "agent answered without sources"
assert data["thread_id"], data
assert all(s["domain"] == "'"${DOMAIN}"'" for s in data["sources"]), data["sources"]
print("   domains=%s sources=%d" % (data["domains_searched"], len(data["sources"])))
print("   answer: %s" % data["answer"][:160])
'

say "streaming"
curl -fsS -N -X POST "${BASE_URL}/agent/chat/stream" -H "${AUTH}" -H 'Content-Type: application/json' \
  -d "{\"message\": \"How much paid leave do employees accrue?\", \"domains\": [\"${DOMAIN}\"]}" \
  | grep -q '"type": "done"' || fail "stream did not finish with a done event"

say "stats"
curl -fsS "${BASE_URL}/kb/${DOMAIN}/stats" -H "${AUTH}" | "${PYTHON}" -c '
import json,sys
data = json.load(sys.stdin)
assert data["vector_count"] > 0, data
print("   store=%s vectors=%d model=%s dim=%d" % (
    data["store"], data["vector_count"], data["embedding_model"], data["embedding_dimension"]))
'

printf '\033[1;32mSMOKE TEST PASSED\033[0m\n'
