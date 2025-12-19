import torch
import json
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import torch.nn as nn
from model import BiLSTM_CRF

# 모델 로드 함수 (huggingface hub 에서 다운로드)
def load_model_from_hub(repo_id, device):
    print(f">>> HuggingFace Hub에서 모델 다운로드 중... {repo_id}")

    # 1. 파일 다운로드
    model_path = hf_hub_download(repo_id=repo_id, filename="klue_ner_bi_lstm_crf.bin")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    # 2.config 로드
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 3. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # 4. 모델 뼈대 생성
    # config에 저장된 tag_to_ix(label2id)를 가져옴
    tag_to_ix = config["tag_to_ix"]
    label2id = {int(v): k for k, v in tag_to_ix.items()}

    model = BiLSTM_CRF(
        vocab_size=config['vocab_size'],
        tag_to_ix=tag_to_ix,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )

    # 5. 가중치(State Dict) 로드
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(">>> 모델 로드 완료!")
    return model, tokenizer, label2id

def predict_ner(text, model, tokenizer, label2id, device):
    # 전처리
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)

    # attention_mask는 0,1로 되어 있으므로 Boolean(True/False)로 변환
    mask = inputs["attention_mask"].bool().to(device)

    # Offset Mapping 추출 (후처리에 사용)
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

    # 모델 추론
    with torch.no_grad():
        # returns List[List[int]]
        predictions = model.decode(input_ids, mask)
        pred_tags_ids = predictions[0]
    
    # 결과 매핑 (ID -> Label) & 특수 토큰 제외
    valid_tokens = []

    # input_ids[0]과 pred_tags_ids의 길이는 같습니다 (마스크 된 부분 제외)
    # 하지만 tokenizer는 [CLS], [SEP]을 포함하므로 인덱스를 주의해야 합니다.

    for idx, (offset, tag_id) in enumerate(zip(offset_mapping, pred_tags_ids)):
        start, end = offset

        # [CLS], [SEP] 등은 offset이 (0, 0)입니다. 이는 건너뜁니다.
        if start != end:
            label = label2id[tag_id]
            valid_tokens.append((start, end, label))
    return format_result(text, valid_tokens)

def format_result(original_text, valid_tokens):
    """
    BIO 태그를 파싱하여 <단어:태그>형태로 변환
    """

    result_text = ""
    processed_idx = 0

    current_entity = None
    entities = []

    for start, end, label in valid_tokens:
        if label.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"start":start, "end":end, "label":label.split("-")[1]}
        elif label.startswith("I-"):
            if current_entity and label.split("-")[1] == current_entity['label']:
                current_entity['end'] = end
            else:
                if current_entity: entities.append(current_entity)
                current_entity = {"start":start, "end":end, "label":label.split("-")[1]}
        else: # 'O' 태그
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity: entities.append(current_entity)

    # 문자열 조립
    for entity in entities:
        start, end, label = entity['start'], entity['end'], entity['label']
        result_text += original_text[processed_idx:start]
        result_text += f"<{original_text[start:end]}:{label}>"
        processed_idx = end
    
    result_text += original_text[processed_idx:]
    return result_text

if __name__ == "__main__":
    # 설정
    repo_id = "본인의 허깅페이스 이름/klue-ner-bi-lstm-crf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model, tokenizer, label2id = load_model_from_hub(repo_id, device)

    # 테스트 문장
    test_sentences = [
        "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.",
        "손흥민은 1992년 7월 8일 춘천에서 태어났다.",
        "SK하이닉스는 경기도 이천에 본사를 두고 있다."
    ]

    print("\n" + "="*60)
    for text in test_sentences:
        result = predict_ner(text, model, tokenizer, label2id, device)
        print(f"입력: {text}")
        print(f"결과: {result}")
        print("-" * 60)