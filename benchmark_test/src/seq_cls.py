import torch
import numpy as np
import pandas as pd
import os

def result_output_seq_classification(
    trainer,
    tokenized_datasets,
    tokenizer,
    id2label,
    target_split="validation",
    output_filename='seq_classification_results.csv'
):
    """
    Sequence Classificationの予測結果を出力する関数
    Args:
        trainer: トレーナーオブジェクト
        tokenized_datasets: トークナイズされたデータセット (Dict)
        tokenizer: トークナイザー
        id2label: ラベルIDからラベル名へのマッピング辞書
        target_split (str): 予測を行うデータセットのキー ("validation" や "test")
        output_filename (str): 出力CSVファイル名
    Returns:
        pd.DataFrame: 予測結果を含むデータフレーム
    """
    print(f"Generating predictions for {target_split} set...")
    
    predictions, labels, _ = trainer.predict(tokenized_datasets[target_split])

    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=1).numpy()
    
    pred_label_ids = np.argmax(predictions, axis=1)

    ordered_label_names = [id2label[i] for i in range(len(id2label))]
    all_results_list = []

    num_samples = len(tokenized_datasets[target_split])
    
    for sample_idx in range(num_samples):
        input_ids = tokenized_datasets[target_split][sample_idx]["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        true_label_id = labels[sample_idx]
        true_label = id2label.get(true_label_id, "Unknown")
        
        pred_label_id = pred_label_ids[sample_idx]
        pred_label = id2label.get(pred_label_id, "Error")

        row_data = {
            'Text': text,              
            'Predicted': pred_label,   
            'True_Label': true_label,  
            'Is_Correct': pred_label == true_label 
        }

        for label_id, label_name in enumerate(ordered_label_names):
            prob = probabilities[sample_idx, label_id]
            row_data[f'P({label_name})'] = prob
        
        all_results_list.append(row_data)

    df = pd.DataFrame(all_results_list)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_filename}")

    return df