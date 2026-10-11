# 言語班の共通環境 (nlp-envs)

**今の版: 2026.10.0** (2026-10-11) ／ 変更の記録: [CHANGELOG.md](CHANGELOG.md) ／ 作った SIF の記録: [RELEASES.md](RELEASES.md)

言語班のみんなが同じ版のパッケージで実験できるように、よく使う環境を4種類用意し、
x86 の計算機用と grace (ARM の CPU) 用の2つずつ、合わせて **8つ** を版つきで管理します。
どれも **uv** で作り、uv で使えます。

> 2026-10-11 時点で、定義 (このフォルダ) はできていますが、**SIF はまだ1つも作っていません** (7章)。

---

## 1. 何が解決するか

これまでは、プロジェクトごとに `pyproject.toml` と `uv.lock` を作っていたので、同じ言語班の中でも版がばらばらでした
(2026-10 の調べ: torch 2.9〜2.11、transformers 4.57〜5.12、vLLM 0.22〜0.23)。
組み合わせを各自で解くと、気づかないうちに古い版に戻ることもありました
(vLLM と同じ環境で `datasets` が 2020 年の 1.1.1 に戻っていたプロジェクトがあった)。

ここでは、組み合わせを一度だけ解いて (`uv.lock`)、x86 と ARM の両方で同じ版を使えるようにします。

## 2. 8つの環境

| 環境 | 使い道 | x86 の計算機用 (creator・david・andre・hiegm5・preproc01・micgm1) | grace 用 (ARM) |
| :--- | :--- | :--- | :--- |
| **nlp** | 前処理・BERT 系の学習と評価・データ処理 | `nlp-2026.10.0-x86_64.sif` | `nlp-2026.10.0-aarch64.sif` |
| **llm** | vLLM での推論・LLM での判定・API | `llm-2026.10.0-x86_64.sif` | `llm-2026.10.0-aarch64.sif` |
| **train** | LoRA などの軽い学習 | `train-2026.10.0-x86_64.sif` | `train-2026.10.0-aarch64.sif` |
| **es** | Elasticsearch (検索のサーバー) | `es-2026.10.0-x86_64.sif` | `es-2026.10.0-aarch64.sif` |

- 名前は `<環境>-<版>-<CPU>.sif` です。**x86 用の SIF は grace では動きません** (逆も同じ)。`uname -m` で CPU を確かめます。
- `llm` と `train` は、`nlp` の中身を全部含みます。

## 3. 中身と主な版

| | 版 | 入っている環境 |
| :--- | :--- | :--- |
| Python | 3.12 | 全部 |
| CUDA (コンテナの土台) | 13.0 (`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`) | nlp・llm・train |
| torch | 2.13.0 (CUDA 13) | nlp・llm・train |
| transformers | 5.17.0 | nlp・llm・train |
| datasets / accelerate / tokenizers | 5.1.0 / 1.15.0 / 0.23.3 | nlp・llm・train |
| numpy / pandas / polars / duckdb | 2.3.5 / 3.0.6 / `uv.lock` のとおり | nlp・llm・train |
| spaCy | 3.8.16 (scispaCy は入っていない。8章) | nlp・llm・train |
| elasticsearch (クライアント) | 8.19.3 | nlp・llm・train |
| vLLM | 0.31.0 | llm |
| openai / anthropic / tiktoken | `uv.lock` のとおり | llm |
| peft / trl / bitsandbytes / wandb | 0.21.2 / 1.15.0 / 0.50.2 / `uv.lock` のとおり | train |
| Elasticsearch (サーバー) | 8.19.23 | es |

全部の版は [`uv.lock`](uv.lock) にあります (307 個)。何を入れるかは [`pyproject.toml`](pyproject.toml) に書いています。

## 4. 使い方

### 4.1 SIF をそのまま使う (一番かんたん)

```bash
SIF=<SIF を置いた場所>/llm-2026.10.0-$(uname -m).sif
apptainer exec --nv "$SIF" python my_script.py
```

- `--nv` は GPU を使うための指定です。コンテナの中では `python` が、この環境の Python を指します。
- ジョブの中で使います (ログインノードで重い計算はしない)。

### 4.2 SIF を使わずに、uv で同じ環境を作る

grace のように SIF を使いにくい機械や、手元で試すときです。CPU に合った版が自動で入ります。

```bash
git clone <nlp-util のリポジトリ> && cd nlp-util/envs
uv sync --frozen                  # nlp
uv sync --frozen --extra llm      # llm
uv sync --frozen --extra train    # train
```

- `--frozen` は「`uv.lock` のとおりに入れる。版を選び直さない」という意味です。
- できた `.venv` は数GB〜十数GB あります。ファイルサーバーの上だと読み込みが遅いので、なるべく 4.1 を使います。
- NVIDIA のドライバが 580 以上の機械で動きます (CUDA 13 のため)。preproc01 はドライバが入っていないので、GPU は使えません。

### 4.3 自分のプロジェクトで、パッケージを足して使う (uv)

共通の版をそのまま引き継ぎ、足りないものだけを足します。

```bash
cd my_project
cp <nlp-util>/envs/pyproject.toml <nlp-util>/envs/uv.lock .
# pyproject.toml の name をプロジェクトの名前に変え、[tool.nlp-envs] の表は消す
apptainer exec --nv llm-2026.10.0-$(uname -m).sif uv add <足すパッケージ>
apptainer exec --nv llm-2026.10.0-$(uname -m).sif uv sync --extra llm
```

- `uv add` は、`uv.lock` にある版をできるだけそのまま使って、足したものだけを解きます。共通の版から外れたら、`uv.lock` の差分に出ます。
- ちょっとだけ足したいときは、SIF の環境の上に小さな venv を重ねることもできます (SIF の中身は書き換えられないため):

  ```bash
  apptainer exec --nv "$SIF" uv venv --python /opt/nlp-envs/.venv/bin/python --system-site-packages .venv-extra
  apptainer exec --nv "$SIF" bash -c 'source .venv-extra/bin/activate && uv pip install <パッケージ> && python my_script.py'
  ```

### 4.4 Elasticsearch

```bash
bash envs/es/run_es.sh es-2026.10.0-$(uname -m).sif <データのフォルダ> 9200 4g
```

- ジョブの中で、その機械の中だけ (`127.0.0.1`) で動かします。Python からは `Elasticsearch("http://127.0.0.1:9200")` でつなぎます。
- **データのフォルダの注意**: 今までの SIF (8.13.2) で作ったデータを 8.19 で開くと、データが新しい形に書き換わり、8.13 では二度と開けなくなります。
  今のデータを使い続けるなら、先にコピーを取ってから開きます。

## 5. 版の決まり

- 版は **`年.月.番号`** です (例 `2026.10.0`)。同じ月に直したら番号を増やします (`2026.10.1`)。
- 版は [`pyproject.toml`](pyproject.toml) の `project.version` の1か所だけに書きます。SIF の名前・中のラベル・記録はここから作ります。
- **一度作った SIF は上書きしません。** 中身を変えるときは、必ず版を上げて別の名前で作ります。古い版は、使っている実験が無くなるまで残します。
- 作った SIF は [RELEASES.md](RELEASES.md) に、SHA-256 (中身が同じかを確かめる指紋) と一緒に記録します。
- Git のタグは `nlp-envs-<版>` (例 `nlp-envs-2026.10.0`) を、その版の SIF を全部作ってテストが通ったコミットに付けます。
- 実験の記録には、使った SIF の名前 (版と CPU 入り) を残します。

## 6. 更新のしかた

1. `pyproject.toml` を直す (パッケージを足す・版を変える)。版も上げ、[CHANGELOG.md](CHANGELOG.md) に何を変えたかを書く。
2. 組み合わせを解き直す (版の組み合わせを決めるだけで、パッケージは入れない):

   ```bash
   cd envs && apptainer exec <どれかの SIF> uv lock
   ```

3. 静的なテストを通す: `python3 -m unittest discover -s envs/tests` (リポジトリの一番上で)。
   - x86 と ARM のどちらかで入らない版、ソースからの組み立てが新しく必要になったパッケージ、`pyproject.toml` と `uv.lock` の食い違いを見つけます。
4. コミットしてから、x86 と ARM の両方で作り (7章)、テストし、RELEASES.md に記録して、タグを付けます。

## 7. 作る

作るのは Slurm のジョブです。エージェント (Claude など) はジョブを出せない決まりなので、**人が出します**。
どちらも、コミットしてから出します (変更が残っていると止まります)。

**x86 用 (micgm1 で)**:

```bash
cd <nlp-util のフォルダ>
sbatch -p large-micgm1 envs/build.sbatch                   # 4つの SIF を作る (最大 4 時間)
sbatch -p small-micgm1 -d afterok:<上のジョブ番号> envs/test.sbatch   # GPU で確かめる
```

- 置き場所は、既定でリポジトリの外の `../../images/` (`--out` で変えられます)。作業用の一時ファイルは `/scratch` があればそこ、無ければ置き場所の下の `.work/` です。

**grace 用 (grace02 で。parent から)**:

```bash
cd /workspace/filesrv01/<自分のユーザー名>/nlp-util          # nlp-util を filesrv01 の上に置いて、同じコミットにする
sbatch -p large-grace-o -w grace02 envs/build.sbatch --out /workspace/filesrv01/<自分>/nlp-envs/images
sbatch -p small-grace-o -w grace02 -d afterok:<上のジョブ番号> envs/test.sbatch --out /workspace/filesrv01/<自分>/nlp-envs/images
```

- 今のクラスタは、`-o` のパーティションには filesrv01 の上のフォルダから出す決まりです (grace02 は `-o`)。
- grace01 は修理中です (2026-10)。

**作った後**: 各 SIF の横の `.sha256` の値を [RELEASES.md](RELEASES.md) に書き、`.test.log` が全部 PASS であることを確かめます。

## 8. 注意

- **ドライバ**: CUDA 13 なので、NVIDIA のドライバ 580 以上が必要です (2026-09 の機器一覧では、ドライバのある機械は全部 580 以上)。
- **GB10 (grace)**: torch は sm_120 の部品で GB10 (sm_121) を動かします。vLLM は 0.19 から GB10 に対応しています。動くかどうかは 7章のテストで確かめます。
- **NVIDIA の CUTLASS のライブラリ**: `nvidia-cutlass-dsl` の決まりで、CUDA 12 用と 13 用の両方が入ります (実行時に合う方が使われる)。
- **Elasticsearch のデータ**: 4.4 の注意を読んでください。
- **scispaCy は入っていません**: どの版も numpy 1 が要り (0.5.5 は spaCy 3.7 を、0.6.2 は numpy<2 を求める)、numpy 2 が要る vLLM と同じ環境に入れられないためです。
  spaCy 3.7 と numpy 2 の組み合わせは、`uv lock` は通っても `import spacy` で止まります (`numpy.dtype size changed`)。
  text_preprocess_with_llm と claim-KG の `uv.lock` もこの組み合わせです (2026-10 時点ではどちらも spaCy を import していないので、表に出ていない)。
  scispaCy が要るときは、相談してください (numpy 1 の別の環境を作るなど)。

## 9. まだのこと

- SIF をまだ作っていません (7章)。作ったら、RELEASES.md に記録します。
- **ほかの人が使える置き場所**: 作った SIF をどこに置くかは未定です。案は、クラスタの各ノードの `/opt/lab/images` に配る (研究室のクラスタのキットの D-SIF-1)、
  またはファイルサーバーの共有フォルダ (filesrv01 と filesrv02 の両方。今はノードが片方しか見ないため) です。
- 言語班のほかの人のプロジェクトの環境は、まだ合わせていません (2026-10 は大戸のプロジェクト 4 つから作った)。
