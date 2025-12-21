import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

# 1. 설정 및 모델 로드
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
#ADAPTER_REPO_ID = "본인 허깅페이스 계정 이름/EXAONE-3.0-7.8B-KLUE-NER-LoRA"

print(">>> 모델 로드 중 (잠시만 기다려 주세요)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)

#model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_ID)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = (
    "당신은 유능한 개체명 인식기(NER)입니다. "
    "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
    "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
    "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
)

# 2. 추론 함수 정의
def predict_ner(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id = tokenizer.eos_token_id
        )
    response = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response, skip_special_tokens=True)

    # JSON 파싱 시도
    try:
        json_obj = json.loads(decoded_output)
        return json_obj, decoded_output
    except:
        return None, decoded_output

# 3. 사용자 입력 루프
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  sLLM 개체명 인식기 (Type 'exit' to quit)")
    print("="*50)

    # 예제 문장 자동 실행
    example_text = "손흥민은 1992년 7월 8일 춘천에서 태어났다."
    print(f"\n[예제] 입력: {example_text}")
    result, raw_str = predict_ner(example_text)
    print(f"[예제] 결과: {raw_str}")

    while True:
        user_input = input("\n문장을 입력하세요: ")
        if user_input.lower() in ["exit", "quit", "종료"]:
            break

        if not user_input.strip():
            continue

        result_json, raw_text = predict_ner(user_input)

        print(f"Running...")
        print("-" * 30)
        if result_json:
            # 예쁘게 출력
            print(json.dumps(result_json, indent=4, ensure_ascii=False))
        else:
            # 파싱 실패 시 원본 출력
            print(f"Raw Output: {raw_text}")
            print("(JSON 파싱에 실패했습니다.)")
        print("-" * 30)
