import torch
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

# 1. 저장된 모델과 토크나이저 불러오기 (경로 주의)
#model_path = "./final_ner_model"  # 학습 후 저장한 경로

model_id = "Laseung/klue-bert-base-klue-ner-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# 2. 데이터셋 및 평가 지표 준비 (학습 때와 동일해야 함)
dataset = load_dataset("klue", "ner")
metric = evaluate.load("seqeval")
label_list = dataset["train"].features["ner_tags"].feature.names

# 3. 전처리 함수 (학습 때 사용한 함수와 100% 동일해야 함)
# (앞서 작성했던 offset_mapping을 사용하는 함수를 그대로 가져옵니다)
def tokenize_and_align_labels(examples):
    raw_inputs = ["".join(x) for x in examples["tokens"]]
    tokenized_inputs = tokenizer(
        raw_inputs,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
        encoded_labels = []
        for offset in offset_mapping:
            start, end = offset
            if start == end: 
                encoded_labels.append(-100)
                continue
            origin_char_idx = start
            if origin_char_idx < len(doc_tags):
                encoded_labels.append(doc_tags[origin_char_idx])
            else:
                encoded_labels.append(-100)
        labels.append(encoded_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Validation 데이터셋 전처리
eval_dataset = dataset["validation"].map(tokenize_and_align_labels, batched=True)

# 4. 평가 함수 정의
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. Trainer를 이용해 평가 수행
trainer = Trainer(
    model=model,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print(">>> 검증 시작...")
metrics = trainer.evaluate()

print(f"\n>>> [검증 결과] F1-score: {metrics['eval_f1']:.4f}")
print(f">>> (학습 로그의 Best F1과 비교해 보세요)")