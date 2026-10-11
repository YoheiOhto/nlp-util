#!/usr/bin/env bash
# ==============================================================================
# nlp-envs の Elasticsearch を1台 (single-node) で動かす
# ==============================================================================
# 使い方 (ジョブの中で。終わるまで前で動き続ける。止めるときは Ctrl+C か、ジョブを終える):
#   bash envs/es/run_es.sh <es-....sif> <データのフォルダ> [ポート (既定 9200)] [ヒープ (既定 4g)]
#
# - 同じ機械の中からだけ使う (127.0.0.1 で待ち受ける)。別の機械から使う設定にはしない。
#   外から届くようにすると、Elasticsearch が本番向けの確認 (vm.max_map_count など、機械の設定) を求めるため。
# - セキュリティ (ログインと TLS) は切っている。研究室の中の、ジョブの中だけで使う前提。
# - データのフォルダは、ほかの版の Elasticsearch で作ったものを渡さない。新しい版で開くと書き換わり、
#   古い版では二度と開けなくなる (README.md の「Elasticsearch の注意」)。
# ==============================================================================
set -euo pipefail

sif="${1:?usage: run_es.sh <es sif> <data dir> [port] [heap]}"
data="${2:?usage: run_es.sh <es sif> <data dir> [port] [heap]}"
port="${3:-9200}"
heap="${4:-4g}"

mkdir -p "$data/data" "$data/logs"
ulimit -n 65535 2>/dev/null || echo "WARN: could not raise the open-file limit (ulimit -n)" >&2

exec apptainer exec \
    --cleanenv \
    --writable-tmpfs \
    --bind "$data/data:/usr/share/elasticsearch/data" \
    --bind "$data/logs:/usr/share/elasticsearch/logs" \
    --env "discovery.type=single-node" \
    --env "network.host=127.0.0.1" \
    --env "http.port=${port}" \
    --env "xpack.security.enabled=false" \
    --env "ES_JAVA_OPTS=-Xms${heap} -Xmx${heap}" \
    "$sif" \
    /usr/local/bin/docker-entrypoint.sh eswrapper
