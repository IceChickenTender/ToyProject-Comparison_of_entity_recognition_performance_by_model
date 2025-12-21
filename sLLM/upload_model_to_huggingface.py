import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# 1. 설정
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA" # 학습 결과가 저장된 로컬 폴더 이름
HF_USERNAME = "본인의_허깅페이스_계정명" # 예: wonmin
HF_TOKEN = "본인의_WRITE_권한_토큰"

print(">>> 모델 로드 준비 중...")

# 2. [수정 1] 오타 수정 (dtye -> dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 베이스 모델 로드 (껍데기)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. 학습된 LoRA 어댑터 결합
# PeftModel을 로드하면 push_to_hub 실행 시 자동으로 '어댑터 파일'만 업로드합니다.
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)

# 4. [개선] 토크나이저를 '로컬 학습 폴더'에서 로드
# 학습할 때 저장해둔 설정을 그대로 올리는 것이 가장 안전합니다.
try:
    print(f">>> 로컬 폴더({ADAPTER_MODEL_PATH})에서 토크나이저 로드 시도...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True)
except:
    print(">>> 로컬 토크나이저 로드 실패, 베이스 모델에서 로드합니다.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# 5. 허깅페이스 로그인 및 업로드
print(">>> Hugging Face 로그인...")
login(token=HF_TOKEN)

# [수정 2] 레포지토리 ID 포맷 수정 (슬래시 추가)
repo_id = f"{HF_USERNAME}/{ADAPTER_MODEL_PATH}"
print(f">>> 업로드 시작: {repo_id}")

# 어댑터 업로드 (adapter_model.safetensors, adapter_config.json 등)
model.push_to_hub(repo_id, use_temp_dir=False)

# 토크나이저 업로드
tokenizer.push_to_hub(repo_id, use_temp_dir=False)

print(">>> 업로드 완료! 허깅페이스 웹사이트에서 확인하세요.")