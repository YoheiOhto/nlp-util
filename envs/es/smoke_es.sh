#!/usr/bin/env bash
# ==============================================================================
# nlp-envs の Elasticsearch の SIF を確かめる (test.sh が呼ぶ)
# ==============================================================================
# 使い方: bash envs/es/smoke_es.sh <es-....sif>
# 一時フォルダで起動し、1件入れて検索できたら止める。データは残さない。
# ==============================================================================
set -euo pipefail

sif="${1:?usage: smoke_es.sh <es sif>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/nlp-envs-es.XXXXXX")"
port=$((19200 + RANDOM % 500))
url="http://127.0.0.1:${port}"

cleanup() {
    [ -n "${es_pid:-}" ] && kill "$es_pid" 2>/dev/null && wait "$es_pid" 2>/dev/null
    rm -rf "$tmp"
}
trap cleanup EXIT

bash "$here/run_es.sh" "$sif" "$tmp" "$port" 1g > "$tmp/es.out" 2>&1 &
es_pid=$!

for _ in $(seq 1 90); do
    if curl -fsS "$url/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$es_pid" 2>/dev/null; then
        echo "FAIL es: Elasticsearch exited during start-up"; tail -40 "$tmp/es.out"; exit 1
    fi
    sleep 2
done

version="$(curl -fsS "$url" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"]["number"])')" \
    || { echo "FAIL es: no answer on ${url} within 180 s"; tail -40 "$tmp/es.out"; exit 1; }
curl -fsS -X PUT "$url/smoke/_doc/1?refresh=true" -H 'Content-Type: application/json' \
    -d '{"text": "aspirin inhibits cyclooxygenase"}' >/dev/null
hits="$(curl -fsS "$url/smoke/_search?q=text:aspirin" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hits"]["total"]["value"])')"
if [ "$hits" = "1" ]; then
    echo "PASS es: Elasticsearch ${version} indexed and found a document on ${url}"
else
    echo "FAIL es: search returned ${hits} hits (expected 1)"; exit 1
fi
