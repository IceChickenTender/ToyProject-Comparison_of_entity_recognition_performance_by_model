import torch
import json
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. 설정 및 정보
repo_id = "본인의 허깅페이스 이름/klue-ner-bi-lstm-crf"
token = "본인의 허깅페이스 token 값"
save_dir = "./upload_pack"

# 폴더 생성
os.makedirs(save_dir, exist_ok=True)

# 라벨 맵핑
dataset = load_dataset("klue", "ner")
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}

# 2. 필수 파일 준비

# Config 파일 저장 (모델 구조 복원을 위해 필수)
config = {
    "vocab_size": 32000, # tokenizer.vocb_size
    "embedding_dim": 768, # 사용한 값
    "hidden_dim": 256, # 사용한 값
    "tag_to_ix" : label2id, # 라벨 맵핑 정보
    "model_type": "BiLSTM_CRF" # 식별용
}

with open(f"{save_dir}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# 학습된 모델이 메모리에 있다면:
# torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin") 
# *Tip: 허깅페이스는 보통 .pth 대신 pytorch_model.bin 이라는 이름을 씁니다.

# 모델 가중치 복사 혹은 저장
import shutil
shutil.copy("./models/klue_ner_bi_lstm_crf.pth", f"{save_dir}/klue_ner_bi_lstm_crf.bin")

# 토크나이저 저장
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer.save_pretrained(save_dir)

# 3. 허깅페이스 허브에 업로드
print(f">>> Uploading to {repo_id}...")

try:

    #레포지토리 생성
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # 폴더 내 모든 파일 업로드
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print(">>> 업로드 성공! 아래 주소에서 확인해보세요.")
    print(f"https://huggingface.co/{repo_id}")
except Exception as e:
    print(f">>> 업로드 실패: {e}")