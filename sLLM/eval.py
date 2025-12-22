import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import data_processing as pr
import json

# 1. 설정

BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
#ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
ADAPTER_REPO_ID = "Laseung/EXAONE-3.0-7.8B-KLUE-NER-LoRA"
MAX_NEW_TOKENS = 512



# 시스템 프롬프트 (학습 때와 동일해야 함)
# SYSTEM_PROMPT = (
#     "당신은 유능한 개체명 인식기(NER)입니다. "
#     "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
#     "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
#     "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
# )

# 시스템 프롬프트 JSON 파싱 에러가 발생해 Few-shot 적용
# SYSTEM_PROMPT = (
#     "당신은 유능한 개체명 인식기(NER)입니다. "
#     "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요.\n"
#     "태그: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량)\n\n"
#     "[예시]\n"
#     "입력: 손흥민은 2023년 토트넘에서 20골을 넣었다.\n"
#     "출력: {\"PS\": [\"손흥민\"], \"DT\": [\"2023년\"], \"OG\": [\"토트넘\"], \"QT\": [\"20골\"], \"LC\": [], \"TI\": []}\n\n"
#     "이제 다음 문장을 분석하세요."
# )


# 성능 향상을 위한 엄격한 프롬프트 적용 가장 성능이 좋은 프롬프트
SYSTEM_PROMPT = (
    "당신은 엄격한 기준을 가진 개체명 인식기(NER)입니다.\n"
    "입력 문장에서 다음 태그에 해당하는 단어만 정확히 추출하여 JSON으로 출력하세요.\n"
    "태그: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량)\n\n"
    "*** 주의사항 ***\n"
    "1. '탐정', '엔딩크레딧', '시민', '선수' 같은 일반 명사는 절대 추출하지 마세요.\n"
    "2. 문장에 없는 단어를 만들어내거나 요약하지 마세요.\n"
    "3. 날짜나 수량은 문장에 적힌 그대로(단위 포함) 가져오세요.\n"
    "4. 해당하는 개체명이 없으면 빈 리스트를 반환하세요.\n\n"
    "[예시]\n"
    "입력: 명탐정 코난은 추리를 잘한다.\n"
    "출력: {\"PS\": [\"코난\"], \"QT\": [], \"OG\": [], \"LC\": [], \"DT\": [], \"TI\": []}\n"
    "(설명: '명탐정'은 직업이므로 추출하지 않음)\n\n"
    "이제 다음 문장을 분석하세요."
)

# SYSTEM_PROMPT = (
#     "당신은 문맥을 정확하게 파악하는 개체명 인식(NER) 전문가입니다.\n"
#     "주어진 문장에서 아래 가이드라인에 따라 개체명을 추출하여 JSON으로 출력하세요.\n\n"
    
#     "*** 태그별 추출 가이드라인 (중요) ***\n"
#     "- PS (사람): 이름, 직위, 별명 등 (예: 손흥민, 대통령, 안현수)\n"
#     "- LC (장소): 국가, 도시, 장소, 건물명 (예: 한국, 서울, 학교, 톈진항)\n"
#     "- OG (기관): 회사, 학교, 관공서, 팀 이름 (예: 삼성전자, 토트넘, 정부)\n"
#     "- DT (날짜): 연도, 월, 일, 요일, 기간 (예: 2023년, 12월 25일, 오늘, 내년)\n"
#     "- TI (시간): 시각, 시간의 경과/기간 (예: 오후 3시, 20분, 5시간, 밤, 새벽)\n"
#     "- QT (수량): 숫자와 단위가 포함된 수량, 금액, 비율, 순서 (예: 1개, 1000원, 50%, 1위, 3명)\n\n"

#     "*** 주의사항 ***\n"
#     "1. 문장에 있는 단어를 **조사나 띄어쓰기 변경 없이** 그대로 가져오세요.\n"
#     "2. QT(수량)와 TI(시간)는 **숫자뿐만 아니라 뒤에 붙은 단위(개, 원, %, 분, 시간 등)까지** 반드시 포함해야 합니다.\n"
#     "3. 해당하는 개체명이 없으면 빈 리스트 []를 반환하세요.\n"
#     "4. '탐정', '엔딩크레딧', '시민', '선수' 같은 **일반 명사나 직업, 역할은** 절대 추출하지 마세요.\n" 
#     "5. 문장에 없는 단어를 만들어내거나 요약하지 마세요.\n\n"

#     "[예시 1 - 일반]\n"
#     "입력: 손흥민은 2023년 토트넘에서 20골을 넣었다.\n"
#     "출력: {\"PS\": [\"손흥민\"], \"DT\": [\"2023년\"], \"OG\": [\"토트넘\"], \"QT\": [\"20골\"], \"LC\": [], \"TI\": []}\n\n"
    
#     "[예시 2 - 시간과 수량 집중]\n"
#     "입력: 오후 3시 30분에 사과 2박스를 15000원에 샀다.\n"
#     "출력: {\"TI\": [\"오후 3시 30분\"], \"QT\": [\"2박스\", \"15000원\"], \"PS\": [], \"DT\": [], \"OG\": [], \"LC\": []}\n\n"

#     "[예시 3 - 고유 명사에 집중]\n"
#     "입력: 명탐정 코난은 추리를 잘한다.\n"
#     "출력: {\"PS\": [\"코난\"], \"QT\": [], \"OG\": [], \"LC\": [], \"DT\": [], \"TI\": []}\n\n"
    
#     "이제 다음 문장을 분석하세요."
# )

# SYSTEM_PROMPT = (
#     "당신은 문맥에 따라 개체명을 정확히 추출하는 NER 전문가입니다.\n"
#     "입력 문장에서 PS(인물), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량)를 추출하세요.\n\n"
    
#     "*** 규칙 ***\n"
#     "1. 반드시 문장에 적힌 글자 그대로 추출하세요.\n"
#     "2. QT와 TI는 숫자와 단위를 하나로 묶어서 추출하세요. (예: 50%, 3시간)\n"
#     "3. 해당하는 태그가 없으면 []를 출력하세요.\n\n"

#     "*** 출력 형식 ***\n"
#     "{\n"
#     "  \"PS\": [], \"LC\": [], \"OG\": [], \"DT\": [], \"TI\": [], \"QT\": []\n"
#     "}\n\n"

#     "[예시 1]\n"
#     "입력: 안현수는 15일 밤 러시아 소치에서 금메달을 획득했다.\n"
#     "출력: {\"PS\": [\"안현수\"], \"DT\": [\"15일\"], \"TI\": [\"밤\"], \"LC\": [\"러시아 소치\"], \"OG\": [], \"QT\": [\"금메달\"]}\n\n"

#     "[예시 2]\n"
#     "입력: 삼성전자는 내년 말까지 수익률 10% 달성을 목표로 한다.\n"
#     "출력: {\"OG\": [\"삼성전자\"], \"DT\": [\"내년 말\"], \"QT\": [\"10%\"], \"PS\": [], \"LC\": [], \"TI\": []}\n\n"

#     "[예시 3]\n"
#     "입력: 오후 3시에 서울역에서 친구를 만나기로 했다.\n"
#     "출력: {\"TI\": [\"오후 3시\"], \"LC\": [\"서울역\"], \"PS\": [], \"OG\": [], \"DT\": [], \"QT\": []}\n\n"

#     "이제 다음 문장을 분석하세요."
# )

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
#model = PeftModel.from_pretrained(base_model,ADAPTER_MODEL_PATH)
model = PeftModel.from_pretrained(base_model,ADAPTER_REPO_ID)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 4. 평가 실행
print(">>> 데이터셋 로드 및 평가 시작...")
dataset = load_dataset("klue", "ner")
val_data = dataset['validation']

# 평가 시간 단축을 위해 500개만 랜덤 추출하여 평가
small_val_data = val_data.shuffle(seed=42).select(range(100))

# 라벨 정보
label_list = dataset['train'].features['ner_tags'].feature.names
id2label = {i: label for i, label in enumerate(label_list)}

true_labels = []
pred_labels = []

error_count = 0
json_error_result_list = []

# 진행상황 표시
for i in tqdm(range(len(small_val_data))):
    sample = small_val_data[i]
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

    try:
        pred_dict = json.loads(decoded_output)
    except:
        error_count += 1
        json_error_result_list.append({
            "input": input_text,
            "output": decoded_output
        })
        #print(f"JSON Error: {decoded_output}") # 에러 난 문장 확인

    # # [디버깅 코드 추가] ---------------------------------------
    # print(f"\n[Input]: {input_text}")
    # print(f"[Model Output]: {decoded_output}")
    # print(f"[True Tags]: {original_tags}")
    # # --------------------------------------------------------

    # JSON -> BIO 변환
    pred_bio = pr.json_to_bio(decoded_output, tokens)

    # if pred_bio != original_tags: # 틀린 경우만 출력
    #     print("-" * 50)
    #     print(f"입력: {input_text}")
    #     print(f"모델 생성(JSON): {decoded_output}")
        
    #     # 실제 정답 태그가 있는 단어들만 추출해서 보기
    #     true_entities = []
    #     for t, tag in zip(tokens, original_tags):
    #         if tag != 'O': true_entities.append(f"{t}({tag})")
    #     print(f"실제 정답(Target): {true_entities}")
        
    #     # 모델이 예측한 태그가 있는 단어들
    #     pred_entities = []
    #     for t, tag in zip(tokens, pred_bio):
    #         if tag != 'O': pred_entities.append(f"{t}({tag})")
    #     print(f"내 코드의 변환 결과(Bio): {pred_entities}")


    true_labels.append(original_tags)
    pred_labels.append(pred_bio)


print(f"총 샘플 수: {len(small_val_data)}, JSON 파싱 에러: {error_count} 건")

# 점수 계산
print("\n>>> 평가 결과 Report:")
print(classification_report(true_labels, pred_labels))
print(f"F1-Score:{f1_score(true_labels, pred_labels):.4f}")

# for error in json_error_result_list:
#     print(f"\ninput: {error['input']}\n output: {error['output']}\n")
