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

# 止めた Elasticsearch の終了コード (143) を、この確認の結果にしない (ジョブ 2751 は PASS なのに 143 で終わった)
cleanup() {
    local status=$?
    if [ -n "${es_pid:-}" ]; then
        kill "$es_pid" 2>/dev/null || true
        wait "$es_pid" 2>/dev/null || true
    fi
    rm -rf "$tmp"
    exit "$status"
}
trap cleanup EXIT

bash "$here/run_es.sh" "$sif" "$tmp" "$port" 1g > "$tmp/es.out" 2>&1 &
es_pid=$!

# 失敗したときに、どこで待ち受けたか (bound_addresses) と apptainer の警告も見えるようにする
show_log() {
    grep -m5 -E 'invalid var name|bound_addresses' "$tmp/es.out" | cut -c1-400 || true
    tail -40 "$tmp/es.out"
}

ready=0
for _ in $(seq 1 90); do
    if curl -fsS "$url/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$es_pid" 2>/dev/null; then
        echo "FAIL es: Elasticsearch exited during start-up"; show_log; exit 1
    fi
    sleep 2
done
[ "$ready" -eq 1 ] || { echo "FAIL es: no answer on ${url} within 180 s"; show_log; exit 1; }

version="$(curl -fsS "$url" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"]["number"])')"
# run_es.sh の ES_JAVA_OPTS (ヒープ 1g) が届いたか。届かないと、機械のメモリから大きなヒープが選ばれる
heap="$(curl -fsS "$url/_nodes/_local/jvm" | python3 -c 'import json,sys; nodes=json.load(sys.stdin)["nodes"]; print(next(iter(nodes.values()))["jvm"]["mem"]["heap_max_in_bytes"])')"
if [ "$heap" != "1073741824" ]; then
    echo "FAIL es: heap is ${heap} bytes, not the 1g that run_es.sh asked for (ES_JAVA_OPTS did not arrive)"; show_log; exit 1
fi
curl -fsS -X PUT "$url/smoke/_doc/1?refresh=true" -H 'Content-Type: application/json' \
    -d '{"text": "aspirin inhibits cyclooxygenase"}' >/dev/null
hits="$(curl -fsS "$url/smoke/_search?q=text:aspirin" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hits"]["total"]["value"])')"
if [ "$hits" = "1" ]; then
    echo "PASS es: Elasticsearch ${version} on ${url} (heap 1g) indexed and found a document"
else
    echo "FAIL es: search returned ${hits} hits (expected 1)"; exit 1
fi
