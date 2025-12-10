import numpy as np
from datasets import load_dataset
import evaluate
import time
from transformers import(
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
    TrainerCallback
)

# 에포크별 걸린 시간을 측정하기 위한 콜백 함수 정의
class TimeHistoryCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)

# =============================================================================
# 1. 환경 설정 및 하이퍼파라미터 (KLUE 논문 Appendix C 기반)
# =============================================================================
MODEL_ID = "klue/bert-base"
DATASET_ID = "klue"
TASK_NAME = "ner"
SEED = 42  # 재현성을 위한 시드 고정

# 논문 설정값 적용
HYPERPARAMETERS = {
    # 1. 저장 및 로깅 설정
    "output_dir": "./klue-ner-paper-repro", # 학습된 모델 체크포인트와 로그가 저장될 경로
    "logging_steps": 100,                   # 100 step마다 Loss와 Learning Rate를 출력 (진행상황 모니터링)
    "save_strategy": "epoch",               # 매 Epoch이 끝날 때마다 모델을 저장 (중간 저장)
    "eval_strategy": "epoch",         # 매 Epoch이 끝날 때마다 Validation 셋으로 성능(F1) 평가

    # 2. 핵심 학습 파라미터 (논문 재현의 핵심)
    "num_train_epochs": 5,             
    # [논문 설정] 고정값 5. 
    # BERT Fine-tuning은 보통 3~5 epoch 내에 빠르게 수렴합니다. 
    # 너무 길게 잡으면 과적합(Overfitting) 위험이 있습니다.

    "learning_rate": 5e-5,             
    # [논문 설정] 탐색 범위 (1e-5, 3e-5, 5e-5) 중 상한값 선택.
    # NER 같은 Token Classification은 일반 분류보다 조금 높은 학습률(5e-5)에서 
    # Local Minima를 잘 탈출하는 경향이 있습니다.

    "per_device_train_batch_size": 32, 
    # [논문 설정] 탐색 범위 (16, 32).
    # 배치 사이즈 32는 Gradient 추정의 안정성과 학습 속도 간의 균형이 가장 좋습니다.
    # (GPU 메모리가 부족하면 16으로 줄여야 합니다.)

    "per_device_eval_batch_size": 32,  
    # 평가 시 배치 사이즈 (학습 성능에 영향 없음, 속도에만 영향)

    # 3. 최적화 및 규제 (Regularization) - 모델의 일반화 성능 향상
    "warmup_ratio": 0.1,               
    # [논문 설정] 전체 학습 스텝의 10%.
    # "준비 운동"과 같습니다. 학습 초반 10% 동안은 학습률을 0에서 5e-5까지 서서히 올립니다.
    # 이유: 초반부터 높은 학습률을 때려버리면, 잘 학습된 사전학습(Pre-trained) 가중치가 
    # 급격하게 망가지는 것(Catastrophic Forgetting)을 방지하기 위함입니다.

    "weight_decay": 0.01,              
    # [논문 설정] 고정값 0.01 (L2 Regularization).
    # 가중치(Weight) 값이 너무 커지는 것을 억제하는 페널티를 줍니다.
    # 모델이 특정 데이터에만 과도하게 의존하는 것을 막아 과적합을 방지합니다.

    # 4. 모델 선택 및 하드웨어 가속
    "load_best_model_at_end": True,    
    # 학습이 다 끝나면, 저장된 체크포인트 중 가장 성능이 좋았던 모델을 다시 불러옵니다.
    # (마지막 Epoch의 모델이 항상 Best라는 보장이 없기 때문입니다.)

    "metric_for_best_model": "f1",     
    # Best 모델을 선정하는 기준은 'Loss'가 아닌 'F1-score'로 설정합니다.

    "fp16": True,                      
    # [가속 옵션] 16-bit Mixed Precision 사용.
    # 최신 NVIDIA GPU(T4, 3090, A100 등)에서 학습 속도를 2배 가까이 높이고 메모리를 절약합니다.
    # 성능 저하는 거의 없으므로 무조건 켜는 것이 이득입니다.
}

# 시드 고정
set_seed(SEED)

def main():
    print(f">>> 데이터셋 로드 및 전처리 시작 (Model: {MODEL_ID})")

    # =========================================================================
    # 2. 데이터셋 및 토크나이저 로드
    # =========================================================================
    dataset = load_dataset(DATASET_ID, TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 라벨 리스트 및 ID 맵핑 생성
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    # =========================================================================
    # 3. 데이터 전처리 (음절 -> 문장 -> Subword 재정렬 & Offset Mapping)
    # =========================================================================
    def tokenize_and_align_labels(examples):
        # 1) 음절 단위 리스트를 하나의 문장으로 병합
        raw_inputs = ["".join(x) for x in examples["tokens"]]

        # 2) 재토큰화 및 Offset Mapping 반환
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
                
                # 특수 토큰([CLS], [SEP], [PAD]) 처리
                if start == end:
                    encoded_labels.append(-100)
                    continue
                
                # 해당 Subword가 원본의 어느 음절에서 시작했는지 확인
                origin_char_idx = start
                
                # 원본 라벨 매핑
                if origin_char_idx < len(doc_tags):
                    encoded_labels.append(doc_tags[origin_char_idx])
                else:
                    encoded_labels.append(-100)
            
            labels.append(encoded_labels)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 전처리 적용 (불필요한 컬럼 제거)
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # =========================================================================
    # 4. 평가 함수 정의 (seqeval)
    # =========================================================================
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # -100 (특수 토큰) 제외하고 실제 라벨만 복원
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

    # =========================================================================
    # 5. 모델 초기화 및 학습 설정
    # =========================================================================
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(**HYPERPARAMETERS)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 콜백 인스턴스 생성
    time_callback = TimeHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[time_callback]
    )

    # =========================================================================
    # 6. 학습 실행 및 저장
    # =========================================================================
    print(">>> 학습 시작...")
    trainer.train()

    # 7. 결과 분석 및 보고서 출력
    print("\n" + "="*50)
    print(" >>> [상세 분석 보고서] <<<")
    print("="*50)

    # (1) 에포크별 시간 출력
    print(f"\n[1] Epoch별 소요 시간")
    if time_callback.epoch_times:
        for i, duration in enumerate(time_callback.epoch_times):
            print(f" - Epoch {i+1}: {duration:.2f} 초")
        avg_time = sum(time_callback.epoch_times) / len(time_callback.epoch_times)
        print(f" - 평균 소요 시간: {avg_time:.2f} 초")
    else:
        print(" - 시간 데이터가 기록되지 않았습니다.")

    
    # (2) F1-score 분석 (Log History 파싱)
    eval_logs = [log for log in trainer.state.log_history if 'eval_f1' in log]

    if eval_logs:
        f1_scores = [log['eval_f1'] for log in eval_logs]
        epochs = [log['epoch'] for log in eval_logs]
        
        best_f1 = max(f1_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        best_epoch = epochs[f1_scores.index(best_f1)]

        print(f"\n[2] F1-score 분석 (총 {len(f1_scores)}회 평가)")
        print(f" - 전체 평균 F1-score : {avg_f1:.4f}")
        print(f" - 최고(Best) F1-score: {best_f1:.4f} (at Epoch {int(best_epoch)})")
        
        print("\n[상세 기록]")
        for ep, f1 in zip(epochs, f1_scores):
            print(f" - Epoch {int(ep)}: F1 {f1:.4f}")
    else:
        print("\n[!] 평가 로그를 찾을 수 없습니다.")

    # 모델 저장
    print("\n>>> Best 모델 저장 중...")
    trainer.save_model("./final_ner_model_paper_ver")

if __name__ == "__main__":
    main()