# -*- coding: utf-8 -*-
"""
@author: Yohei Ohto
NER実行のための関数群
"""
import evaluate
import numpy as np
import pandas as pd
import torch


def convert_labels_to_ids(example, label2id):
  """
    ラベル文字列をIDに変換する関数
    Args:
        example: データセットの例 (辞書形式 で "ner_tags" を含む)
        label2id: ラベル名からラベルIDへのマッピング辞書
    Returns:
        example: ラベルがIDに変換されたデータセットの例
    How to use:
        raw_datasets = raw_datasets.map(
            lambda x: convert_labels_to_ids(x, label2id),
            batched=False,
        )
  """
  example['ner_tags'] = [label2id[label_str] for label_str in example['ner_tags']]
  return example

def tokenize_and_align_labels(examples, tokenizer):
    """
    トークナイズとラベルのアライメントを行う関数
    Args:
        examples: データセットの例 (辞書形式 で "tokens" と "ner_tags" を含む)
        tokenizer: トークナイザー
    Returns:
        tokenized_inputs: トークナイズされた入力とアライメントされたラベル
    How to use:
        tokenized_datasets = raw_datasets.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def result_output_general(
    trainer,
    tokenized_datasets,
    tokenizer,
    id2label,
    num_samples_to_process=20,
    output_filename='ner_predictions_general_output.csv'
):
    """
    NERの予測結果を一般的な形式で出力する関数
    Args:
        trainer: トレーナーオブジェクト
        tokenized_datasets: トークナイズされたデータセット
        tokenizer: トークナイザー
        id2label: ラベルIDからラベル名へのマッピング辞書
        num_samples_to_process (int): 処理するサンプル数の上限
        output_filename (str): 出力CSVファイル名
    Returns:
        pd.DataFrame: 予測結果を含むデータフレーム
    """
    predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])

    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=2).numpy()
    pred_label_ids = np.argmax(predictions, axis=2)

    ordered_label_names = [id2label[i] for i in range(len(id2label))]

    all_results_list = []


    for sample_idx in range(num_samples_to_process):
        input_ids = tokenized_datasets["validation"][sample_idx]["input_ids"]
        true_label_ids = labels[sample_idx]
        last_true_label = "O"

        for i in range(len(input_ids)):
            token = tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
            if token in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                continue

            pred_label = id2label.get(pred_label_ids[sample_idx, i], "Error")
            if true_label_ids[i] != -100:
                true_label = id2label[true_label_ids[i]]
                last_true_label = true_label
            else:
                true_label = f"({last_true_label})"

            row_data = {
                'Token': token,
                'Predicted': pred_label,
                'True_Label': true_label
            }
            for label_id, label_name in enumerate(ordered_label_names):
                prob = probabilities[sample_idx, i, label_id]
                row_data[f'P({label_name})'] = prob
            
            all_results_list.append(row_data)

        separator_row = {'Token': '--- SENTENCE END ---', 'Predicted': '', 'True_Label': ''}
        for label_name in ordered_label_names:
            separator_row[f'P({label_name})'] = np.nan
        all_results_list.append(separator_row)

    df = pd.DataFrame(all_results_list)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    return df