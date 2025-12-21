import json
import os
from tqdm import tqdm
from datasets import load_dataset
import re

def bio_to_json(tokens, tags, id2label):
    """
    BIO 태그 리스트를  파시앟여 개체명 딕셔너리(JSON)로 변환하는 핵심 함수
    예: tokens=['손','흥','민','은'], tags=[B-PS, I-PS, I-PS, O] -> {'PS': ['손흥민']}
    """
    entity_dict = {}
    current_entity_text = []
    current_entity_label = None

    for token, tag_id in zip(tokens, tags):
        tag_label = id2label[tag_id]

        # 1. 'B-' 태그를 만났을 때
        if tag_label.startswith("B-"):
            # 이전 개체명이 있었다면 저장
            if current_entity_label:
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))

            # 새로운 개체명 시작
            current_entity_label = tag_label.split("-")[1] # "B-PS" -> "PS"
            current_entity_text = [token]
        # 2. 'I-'(Inside) 태그를 만났을 때 (현재 라벨과 같아야 함)
        elif tag_label.startswith("I-") and current_entity_label:
            if tag_label.split("-")[1] == current_entity_label:
                current_entity_text.append(token)
            else:
                # 태그가 꼬인 경우 (B없이 I가 나오거나 다른 I가 나온 경우) - 끊고 새로 시작
                # 여기서는 안전하게 이전 것을 저장하고 초기화
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))
                current_entity_label = None
                current_entity_text = []
        # 3. 'O'(Outside) 태그를 만났을 때
        else:
            if current_entity_label:
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))
                current_entity_label = None
                current_entity_text = []

    # 마지막에 남은 개체명 처리
    if current_entity_label:
        if current_entity_label not in entity_dict:
            entity_dict[current_entity_label] = []
        entity_dict[current_entity_label].append("".join(current_entity_text))

    # 복원된 문장 (KLUE는 tokens 공백 없이 붙이는게 원본에 가까움 필요 시 " ".join 등 조정)
    full_text = "".join(tokens).replace(" ", " ")

    return full_text, json.dumps(entity_dict, ensure_ascii=False)

def create_chat_dataset(dataset_split, id2label):
    formatted_data = []

    # 시스템 프롬프트: 모델에게 역할을 부여하고 출력 형식을 강제함
    system_prompt = (
        "당신은 유능한 개체명 인식기(NER)입니다. "
        "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
        "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
        "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
    )

    for item in tqdm(dataset_split, desc="Processing"):
        tokens = item['tokens']
        ner_tags = item['ner_tags']

        # BIO 태그 -> 텍스트 문장 & JSON 정답 변환
        input_text, output_json = bio_to_json(tokens, ner_tags, id2label)

        # LLM 학습용 Chat Format (OpenAI/HuggingFace 표준)
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_json}
            ]
        }
        formatted_data.append(entry)

    return formatted_data

def preprocessing_data():
    # 1. 데이터셋 로드
    print(">>> KLUE NER 데이터셋 로드")
    dataset = load_dataset("klue", "ner")

    # 2. 라벨 매핑 정보 생성
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}

    # 3. 변환 수행
    train_chat_data = create_chat_dataset(dataset['train'], id2label)
    val_chat_data = create_chat_dataset(dataset['validation'], id2label)

    # 4. JSONL 파일로 저장
    os.makedirs("../data", exist_ok=True)

    with open("../data/train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_chat_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open("../data/validation.jsonl", "w", encoding="utf-8") as f:
        for entry in val_chat_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # 샘플 데이터 1개 확인
    print(json.dumps(train_chat_data[0], indent=4, ensure_ascii=False))

# 3. JSON -> BIO 학습한 모델 평가 시에 사용
def json_to_bio(json_str, tokens):
    """
    모델이 생성한 JSON을 파싱하여 원본 토큰에 맞는 BIO 태그 리스트 생성
    """

    # 1. 초기화 (모두 'O' 태그로 시작)
    predicted_tags = ['O'] * len(tokens)

    try:
        # 모델 출력에서 JSON 부분만 추출 (가끔 잡다한 말을 붙일 수 있음)
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group()
        pred_dict = json.loads(json_str)
    except:
        # 파싱 실패 시 모두 'O' 태그 반환 (형식 불일치 패널티)
        return predicted_tags

    # 2. 텍스트 매칭 및 태그 할당
    # 토큰을 다시 하나의 문자열로 합쳐서 위치를 찾음 (KLUE 토큰 트겅 고려)
    # 주의: 토크나이저 방식에 따라 매핑이 복잡할 수 있으나, 여기선 단순 매칭 시도

    # 원본 문장 재구성 (토큰 offset 매핑을 위해 필요하지만, 약식으로 진행)
    # 실제로는 char_to_tokens 매핑이 필요함. 여기서는 Text 매칭 방식으로 근사.
    token_str_list = tokens # ['손', '흥', '민', '은']

    for label, entity_list in pred_dict.items():
        if not isinstance(entity_list, list): continue

        for entity_text in entity_list:
            # entity_text가 토큰 리스트 안에서 어디에 있는지 찾기(Sliding Window)
            # 예: "손흥민" -> tokens에서 ['손', '흥', '민'] 연속 구간 찾기

            # 토큰들을 합친 임시 문자열 생성
            temp_tokens = "".join(tokens)

            # entity_text의 시작 위치 찾기
            start_idx = temp_tokens.find(entity_text.replae(" ", ""))

            if start_idx == -1: continue

            # Char Index를 Token Index로 변환 (간략화된 로직)
            current_len = 0
            token_start = -1
            token_end = -1

            for i, token in enumerate(tokens):
                token_len = len(token)
                if current_len == start_idx:
                    token_start = i
                if current_len == start_idx + len(entity_text.replace(" ","")) - 1: # 끝 지점
                    token_end = i

                # 범위 안에 있으면
                if token_start != -1 and token_end == -1:
                    # 아직 끝을 못 찾았는데 현재 토큰이 범위 내에 포함되면
                    pass
                elif token_start != -1 and i>= token_start:
                    if current_len + token_len > start_idx + len(entity_text.replace(" ", "")):
                        token_end = i
                current_len += token_len

            # 범위 찾았으면 태그 할당
            if token_start != -1:
                if token_end == -1: token_end = token_start # 1글자짜리

                predicted_tags[token_start] = f"B-{label}"
                for i in range(token_start + 1, token_end + 1):
                    if i < len(predicted_tags):
                        predicted_tags[i] = f"I-{label}"

    return predicted_tags