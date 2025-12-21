import torch
import os
import data_processing as pr
from datasets import load_dataset
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 데이터 전처리를 하지 않았다면 진행
#pr.preprocessing_data()

# 1. 설정 (Configuration)
MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
NEW_MODEL_NAME = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
DATA_PATH_TRAIN = "../data/train.jsonl"
DATA_PATH_VAL = "../data/validation.jsonl"

# 하이퍼파라미터
BATCH_SIZE = 4          # GPU 메모리에 따라 조절 (VRAM 24GB 기준 2~4 추천)
GRAD_ACCUMULATION = 4   # 실제 배치 효과 = BATCH_SIZE * GRAD_ACCUMULATION (여기선 16)
LEARNING_RATE = 2e-4    # QLoRA 표준 학습률
NUM_EPOCHS = 1          # 1 Epoch만 돌아도 데이터가 충분함 (약 2.1만 개)
MAX_SEQ_LENGTH = 1024   # 입력 문장 최대 길이 (메모리 절약을 위해 1024 설정)

# 2. 데이터셋 로드
print(">>> 데이터셋 로드 중...")
if not os.path.isfile("../data/train.jsonl") and not os.path.isfile("../data/validation.jsonl"):
    pr.preprocessing_data()

dataset = load_dataset("json", data_files={"train":DATA_PATH_TRAIN,"validation":DATA_PATH_VAL})

print(">>> 데이터셋 로드 완료!")

# 3. 모델 및 토크나이저 로드(QLoRA 설정)
print(">>> 모델 및 토크나이저 로드 중...")

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 학습 안정성을 위한 설정
model.config.use_cache = False #학습 중엔 캐시 사용을 하지 않음 (Gradient Checkpointing 호환)
model.config.pretraining_tp = 1

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" # SFTTrainer는 right padding을 선호
tokenizer.pad_token = tokenizer.eos_token # EXAONE은 pad 토큰이 명시되지 않을 수 있어 EOS로 대체

# 4. LoRA (PEFT) 설정
# 모델의 모든 레이어를 학습하는 것이 아니라, 일부 레이어에 어댑터를 붙여 학습
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,   # Rank (높을수록 표현력 증가하지만 메모리 사용량 증가)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 모든 선형 레이어 타겟
)

# 모델을 k-bit 학습 준비 상태로 만듦
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 5. Trainer 설정 (SFTTrainer)
# [수정 포인트 1] max_seq_length와 dataset_text_field는 이제 Config 안에 넣어야 합니다.
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    
    # --- [중요] Config 안으로 이동한 파라미터들 ---
    dataset_text_field="messages",  # 데이터셋의 텍스트 컬럼명
    #max_seq_length=MAX_SEQ_LENGTH,  # 최대 길이 설정
    max_length=MAX_SEQ_LENGTH,  # 최대 길이 설정
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    
    # [수정 포인트 2] tokenizer -> processing_class 로 이름 변경 (가장 중요!)
    processing_class=tokenizer, 
    
    args=training_args,
    # max_seq_length=MAX_SEQ_LENGTH,  <-- [삭제] 여기 있으면 에러 납니다!
)

# 6. 학습 시작
print(">>> 학습 시작!")
trainer.train()

print(f">>> 모델 저장 중... {NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
