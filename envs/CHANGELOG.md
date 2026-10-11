# 変更の記録 (nlp-envs)

新しい版を上に足します。版の決まりは [README.md](README.md) の 5章。

## 2026.10.0

2026-10-11。最初の版。

- 4種類 (nlp・llm・train・es) × 2つの CPU (x86_64・aarch64) の8つの環境を、1つの `pyproject.toml` と `uv.lock` で定義した。
- 版は、大戸の言語系のプロジェクト 4 つ (pharma_MLM_bench・text_data_handler・text_preprocess_with_llm・claim-KG。2026-09 の `uv.lock`) を見て決めた。
  - torch 2.13.0 (CUDA 13)、transformers 5.17.0、vLLM 0.31.0 (2026-10-05 の最新)。vLLM が torch 2.13.0 と transformers <5.18 を求めるため、nlp・train も同じ版にそろえた。
  - `datasets` を 4.7 以上にした (今のプロジェクトでは vLLM と組んだときに 1.1.1 まで戻っていた。trl 1.x も 4.7 以上を求める)。
  - scispaCy は今と同じ 0.5.5 (0.6 は numpy<2 を求めるため)。spaCy は 3.7 系。
  - Elasticsearch のサーバーは 8.19.23 (今までの SIF は 8.13.2)。古いデータを開くと戻せない点を README に書いた。
- 土台は `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` (amd64・arm64)、uv 0.12.0、Python 3.12。
- ARM で組み立て済みの部品が無いのは `nmslib-metabrainz` だけ (作るときにコンパイルする)。テスト (`tests/test_envs.py`) で、増えたら気づくようにした。
- まだ SIF は作っていない。
