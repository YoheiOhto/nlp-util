# 変更の記録 (nlp-envs)

新しい版を上に足します。版の決まりは [README.md](README.md) の 5章。

## 2026.10.0

2026-10-11。最初の版。

- 4種類 (nlp・llm・train・es) × 2つの CPU (x86_64・aarch64) の8つの環境を、1つの `pyproject.toml` と `uv.lock` で定義した。
- 版は、大戸の言語系のプロジェクト 4 つ (pharma_MLM_bench・text_data_handler・text_preprocess_with_llm・claim-KG。2026-09 の `uv.lock`) を見て決めた。
  - torch 2.13.0 (CUDA 13)、transformers 5.17.0、vLLM 0.31.0 (2026-10-05 の最新)。vLLM が torch 2.13.0 と transformers <5.18 を求めるため、nlp・train も同じ版にそろえた。
  - `datasets` を 4.7 以上にした (今のプロジェクトでは vLLM と組んだときに 1.1.1 まで戻っていた。trl 1.x も 4.7 以上を求める)。
  - spaCy は 3.8 系 (3.8.16)。最初は今のプロジェクトと同じ spaCy 3.7.5 + scispaCy 0.5.5 にしたが、
    作るときの確認 (ジョブ 2746) で `import spacy` が止まった。spaCy 3.7 の部品 (thinc 8.2.4・blis 0.7.11) は numpy 1 用に組み立てられていて、
    numpy 2 では動かない (thinc 8.2.4 の依存の書き方に numpy<2 が抜けているため、uv が組み合わせとして選んでしまう)。
    vLLM が numpy 2 を求める (opencv-python-headless>=4.13 経由) ので、spaCy を 3.8 に上げた。
  - scispaCy は入れない。どの版も numpy 1 が要る (0.5.5 は spaCy 3.7、0.6.2 は numpy<2)。今のプロジェクトのコードは scispaCy を import していなかった。
  - Elasticsearch のサーバーは 8.19.23 (今までの SIF は 8.13.2)。古いデータを開くと戻せない点を README に書いた。
- 土台は `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` (amd64・arm64)、uv 0.12.0、Python 3.12。
- どのパッケージも、x86_64 と aarch64 の両方に組み立て済みの部品 (wheel) がある (scispaCy を外して、ARM でコンパイルが要る `nmslib-metabrainz` も無くなった)。
  テスト (`tests/test_envs.py`) で、コンパイルが要るものが増えたら気づくようにした。
- 作るときに、利用者のシェルの uv・Python の設定 (`UV_CACHE_DIR` など) をコンテナに持ち込まないようにした (ジョブ 2745 が止まった原因)。
- `es/run_es.sh` は、Elasticsearch の設定 (ポートなど) を `-E` で渡す。apptainer は `http.port` のような点を含む環境変数を捨てるため、
  テストのジョブ 2749 では指定と違うポート (9200) で立ち上がっていた。`es/smoke_es.sh` は、指定したヒープ (1g) が届いたかも確かめる。
- `es/smoke_es.sh` が、確かめ終わって Elasticsearch を止めたときの終了コード (143) を自分の結果にしていたのを直した (ジョブ 2751 は PASS なのに FAIL と出た。2749 の FAIL の終了コードも 1 でなく 143 になっていた)。一時フォルダも残さない。
- x86 用の4つは 2026-10-11 に micgm1 で作り、テストを通った ([RELEASES.md](RELEASES.md))。grace 用はまだ。
