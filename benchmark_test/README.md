# benchmark for Language Model in medical
## TO DO
* 関数化
```markdown
# benchmark for Language Model in medical

## 概要
このディレクトリは医療系言語モデル（主にPubMed系コーパスを対象）に対するベンチマーク実験と実行ノートをまとめたものです。再現可能なノートブックと評価手順を提供します。

## 目的
- 医療ドメインの代表的データセットを用いたモデル評価を行う
- ノートブックを通じて実験の再現性を高める

## 目次（主なノートブック）
- `250717_bc5.ipynb` — BC5CDR の NER 実行例
- `250717_ncbi_desease.ipynb` — NCBI Disease の NER 実行例
- `250718_BLURB.ipynb` — BLURB 相当のセンテンス単位評価の実行例

## 使用データセット
- `ncbi/ncbi_disease`
- `omniquad/BC5CDR-IOB`（疾患と化合物のラベルを含む）

## 必要環境（例）
- Python 3.8+
- Jupyter / JupyterLab
- PyTorch（または対応するバックエンド）、`transformers`, `datasets`, `seqeval` など

セットアップ例:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyterlab transformers datasets torch seqeval
```

## 実行手順（ノートブック）
1. 仮想環境を有効化する
2. `jupyter lab` を起動して該当ノートブックを開く
3. セルを上から順に実行する。データダウンロードやモデルロードに時間がかかるセルはGPU推奨

## 現状の TODO
- ノートブック内処理の関数化（モジュール化して再利用可能にする）
- 結果の自動集約・比較スクリプトの追加

## 参考
- BLURB 実行の参考: https://github.com/michiyasunaga/LinkBERT/tree/main/scripts

---

変更・追加の希望があれば教えてください。コミットしてよければそのままコミット作業を行います。
```