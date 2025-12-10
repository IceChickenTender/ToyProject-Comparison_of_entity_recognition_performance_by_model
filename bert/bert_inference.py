import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
#MODEL_PATH = "./final_ner_model"  # 학습 완료된 모델 경로

# 토크나이저와 모델 불러오기
model_id = "Laseung/klue-bert-base-klue-ner-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# 라벨 맵핑 (모델 config에서 자동으로 가져옵니다)
id2label = model.config.id2label

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==========================================
# 2. 핵심 함수: 문장을 입력받아 태깅된 문장 반환
# ==========================================
def predict_ner(text):
    # 2-1. 입력 텍스트 토크나이징
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True, 
        truncation=True, 
        max_length=128
    )
    
    # offset_mapping은 추론에 불필요하므로 별도로 저장 후 제거
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    
    # 데이터를 GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2-2. 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 2-3. 예측 결과 변환 (Logits -> Tag IDs -> Tag Names)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    pred_tag_ids = predictions[0].cpu().numpy()
    
    # 특수 토큰([CLS], [SEP])을 제외하고 실제 토큰만 필터링
    # offset_mapping을 이용해 실제 글자가 있는 구간만 추출
    valid_tokens = []
    for idx, (offset, tag_id) in enumerate(zip(offset_mapping, pred_tag_ids)):
        start, end = offset
        # [CLS], [SEP] 등은 offset이 (0, 0)
        if start != end:
            label = id2label[tag_id]
            valid_tokens.append((start, end, label))
            
    return decode_ner_tags(text, valid_tokens)

# ==========================================
# 3. 포맷팅 함수: 예측된 태그를 원문에 삽입
# ==========================================
def decode_ner_tags(original_text, valid_tokens):
    """
    valid_tokens: [(start_idx, end_idx, label), ...]
    BIO 태그를 파싱하여 <단어:태그> 형태로 변환
    """
    result_text = ""
    last_idx = 0
    
    # 개체명 묶기 (Chunking) 로직
    current_entity = None # {'start': 0, 'end': 0, 'label': 'QT'}
    
    entities = []
    
    for start, end, label in valid_tokens:
        # BIO 태그 분석
        if label.startswith("B-"):
            # 이전 개체명이 있었다면 저장
            if current_entity:
                entities.append(current_entity)
            
            # 새로운 개체명 시작
            entity_type = label.split("-")[1]
            current_entity = {
                "start": start,
                "end": end,
                "label": entity_type
            }
            
        elif label.startswith("I-"):
            # 현재 진행 중인 개체명이 있고, 타입이 같다면 범위 확장
            if current_entity and label.split("-")[1] == current_entity['label']:
                current_entity['end'] = end
            else:
                # 문법적으로 맞지 않는 I 태그가 나오면(B 없이 I 등), 
                # 이전 개체명 닫고 새로 시작하거나 무시 (여기서는 새로 시작으로 처리)
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label.split("-")[1]
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type
                }
                
        else: # 'O' 태그
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # 마지막에 남은 개체명 처리
    if current_entity:
        entities.append(current_entity)

    # 4. 원본 텍스트에 태그 삽입 (뒤에서부터 작업하면 인덱스가 꼬이지 않음 -> 여기선 순차적으로 작성)
    # 순차적으로 문자열 조립
    processed_idx = 0
    final_output = ""
    
    for entity in entities:
        start = entity['start']
        end = entity['end']
        label = entity['label']
        
        # 개체명 앞부분 붙이기
        final_output += original_text[processed_idx:start]
        
        # 개체명 부분 포맷팅
        entity_text = original_text[start:end]
        final_output += f"<{entity_text}:{label}>"
        
        processed_idx = end
        
    # 남은 뒷부분 붙이기
    final_output += original_text[processed_idx:]
    
    return final_output

# ==========================================
# 4. 실행 테스트
# ==========================================
if __name__ == "__main__":
    test_sentences = [
        "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.",
        "어제 서울 날씨는 맑았지만, 부산은 비가 왔다.",
        "이순신 장군은 조선 시대의 명장이다."
    ]
    
    print("-" * 50)
    for text in test_sentences:
        result = predict_ner(text)
        print(f"입력: {text}")
        print(f"출력: {result}")
        print("-" * 50)