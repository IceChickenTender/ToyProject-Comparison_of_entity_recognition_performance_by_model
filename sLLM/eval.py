import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import data_processing as pr

# 1. 설정

BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
#ADAPTER_REPO_ID = "본인의 허깅페이스 계정 이름/EXAONE-3.0-7.8B-KLUE-NER-LoRA"
MAX_NEW_TOKENS = 512

# 시스템 프롬프트 (학습 때와 동일해야 함)

# 시스템 프롬프트 (학습 때와 동일해야 함)
SYSTEM_PROMPT = (
    "당신은 유능한 개체명 인식기(NER)입니다. "
    "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
    "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
    "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
)

# 2. 모델 로드
print(">>> 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtye=torch.bfloat16,
)

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 학습된 LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model,ADAPTER_MODEL_PATH)
#model = PeftModel.from_pretrained(base_model,ADAPTER_REPO_ID)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 4. 평가 실행
print(">>> 데이터셋 로드 및 평가 시작...")
dataset = load_dataset("klue", "ner")
val_data = dataset['validation']

# 라벨 정보
label_list = dataset['train'].features['ner_tags'].feature.names
id2label = {i: label for i, label in enumerate(label_list)}

true_labels = []
pred_labels = []

# 진행상황 표시
for i in tqdm(range(len(val_data))):
    sample = val_data[i]
    tokens = sample['tokens']
    original_tags = [id2label[tag] for tag in sample['ner_tags']]
    input_text = "".join(tokens).replace(" ", " ") # 원본 문장 복원

    # 프롬프트 구성
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response, skip_special_tokens=True)

    # JSON -> BIO 변환
    pred_bio = pr.json_to_bio(input_text, decoded_output, tokens, id2label)

    true_labels.append(original_tags)
    pred_labels.append(pred_bio)

# 점수 계산
print("\n>>> 평가 결과 Report:")
print(classification_report(true_labels, pred_labels))
print(f"F1-Score:{f1_score(true_labels, pred_labels):.4f}")
