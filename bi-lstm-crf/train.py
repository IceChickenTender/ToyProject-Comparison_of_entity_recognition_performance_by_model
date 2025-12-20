import torch
import time
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from model import BiLSTM_CRF
import torch.optim as optim
from seqeval.metrics import f1_score, classification_report
from transformers import AutoModel
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_align_labels(examples):
    raw_inputs = ["".join(x) for x in examples["tokens"]]
    tokenized_inputs = tokenizer(
        raw_inputs,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length", # 편의상 max_length로 맞추되, 뒤에서 다시 처리 가능
        max_length=128
    )

    labels = []
    for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
        encoded_labels = []
        for offset in offset_mapping:
            start, end = offset
            if start == end:
                encoded_labels.append(-100) # 특수 토큰
                continue
            origin_char_idx = start
            if origin_char_idx < len(doc_tags):
                encoded_labels.append(doc_tags[origin_char_idx])
            else:
                encoded_labels.append(-100)
        labels.append(encoded_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def collate_fn(batch):
    # 1. 데이터 추출
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 2. 패딩(Pytorch 기본 pad_sequence 활용
    # input_ids 패딩(tokenizer.pad_token_id로 채움)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # labels 패딩 (일단 -100이나 'O' 태그로 채움)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # 3. CRF용 마스크 생성
    # labels가 -100인 곳(특수 토큰, 패딩, 서브워드 뒷부분)은 False, 나머지는 True
    # 이렇게 하면 CRF는 해당 지점의 Loss를 계산하지 않음
    mask = (labels_padded != -100)

    # BERT의 첫 번째 토큰은 무조건 [CLS]이므로 인덱스 0입니다.
    # 이 부분의 마스크를 강제로 True로 바꿔줍니다.
    mask[:, 0] = True
    
    # [CLS]의 라벨이 -100이면 에러가 나므로, 의미 없는 태그인 'O' (pad_tag_id)를 할당합니다.
    # (학습에 큰 영향 없이 CRF가 시작점을 잡을 수 있게 해줍니다.)
    labels_padded[:, 0] = pad_tag_id

    # 4. -100 값 치환 (CRF 에러 방지)
    # 마스크가 False인 곳은 계산 안하겠지만, 인덱스 에러 방지를 위해 'O' 태그 ID로 덮어씀

    # 4. 나머지 -100 값 치환
    labels_padded[labels_padded == -100] = pad_tag_id

    return input_ids_padded, labels_padded, mask

# 평가 함수
def evaluate(model, dataloader, id2label):
    model.eval() # 평가 모드로 전환

    true_labels = []
    pred_labels = []

    with torch.no_grad(): # 기울기 계산 끄기
        for batch in dataloader:
            input_ids, tags, mask = batch
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            # 모델 추론
            # CRF의 decode는 Loss가 아니라 Best Tag Sequence(List[List[int]])를 반환합니다.
            predictions = model.decode(input_ids, mask)

            # 4. 정답(Tags)과 예측값(Predictinos)을 문자열로 변환
            # predictions는 가변 길이 리스트이고, tags는 패딩된 텐서이므로 주의해서 매핑
            for pred_seq, true_seq, mask_seq in zip(predictions, tags, mask):
                # pred_seq: [2, 3, 3] (예측된 id 리스트)
                # true_seq: [2, 3, 3, 0, 0] (패딩된 정답 ID 텐서)
                # mask_seq: [True, True, True, False, False]

                # 마스크가 True인 부분만 실제 정답
                valid_true = true_seq[mask_seq].cpu().numpy()

                # ID -> Label 변환 ('2' -> 'B-PS')
                # 예측값 변환
                pred_labels_str = [id2label[tag_id] for tag_id in pred_seq]

                # 정답 변환
                true_labels_str = [id2label[tag_id] for tag_id in valid_true]

                pred_labels.append(pred_labels_str)
                true_labels.append(true_labels_str)

    # seqeval을 이용한 성능 계산
    f1 = f1_score(true_labels, pred_labels)

    print(classification_report(true_labels, pred_labels))

    return f1

model_id = "klue/bert-base"
batch_size = 32

dataset = load_dataset("klue", "ner")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 라벨 맵핑
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

pad_tag_id = label2id['O']

# 전처리 수행
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
# 필요한 컬럼만 남기기 (CRF 모델에는 input_ids와 labels만 필요)
tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
tokenized_datasets = tokenized_datasets.remove_columns(["attention_mask", "offset_mapping"]) # CRF는 별도 마스크 생성 예정
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4, # CPU 병렬 로딩
    pin_memory=True # CPU에서 GPU로의 전송 가속
)

val_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4, # CPU 병렬 로딩
    pin_memory=True # CPU에서 GPU로의 전송 가속
)

# BERT 모델 로드
print(">>> BERT 임베딩 로드 중...")
bert_model = AutoModel.from_pretrained(model_id)

# BERT 임베딩 가중치 추출
bert_embedding_weight = bert_model.embeddings.word_embeddings.weight

vocab_size = tokenizer.vocab_size
tag_to_ix = label2id
EMBEDDING_DIM = 768
HIDDEN_DIM = 256

model = BiLSTM_CRF(
    vocab_size = vocab_size,
    tag_to_ix = tag_to_ix,
    embedding_dim = EMBEDDING_DIM,
    hidden_dim = HIDDEN_DIM
)

# Bi-LSTM-CRF 모델에 BERT 임베딩 가중치 복사
model.word_embeds.weight = nn.Parameter(bert_embedding_weight.clone())

# 선택 사항 임베딩 층 얼리기 vs 학습하기
# True로 설정하면 BERT가 배운 그대로 고정되고, False면 NER 데이터에 맞춰 미세 조정됩니다.
# 데이터가 충분한 KLUE NER의 경우 False(학습 허용)가 성능이 더 좋습니다
model.word_embeds.weight.requires_grad = False

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
path = "./models/klue_ner_bi_lstm_crf.pth"
os.makedirs("./models", exist_ok=True)

best_f1 = 0.0
f1_list = []

print(">>> 학습 시작")
total_start = time.time()
for epoch in range(10):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    
    for batch in train_dataloader:
        input_ids, tags, mask = batch
        input_ids = input_ids.to(device)
        tags = tags.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # 속도 증가를 위해 FP16 연산 적용
        loss = model(input_ids, tags, mask)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    score = evaluate(model, val_dataloader, id2label)
    f1_list.append(score)
    print(f"{epoch+1} Loss : {total_loss/len(train_dataloader)}, F1-score : {score:.4f}")
    if score > best_f1:
        best_f1 = score
        print(f"Best F1-score was changed, Best F1-score is {best_f1:.4f}")
        print("Save Best model...")
        torch.save(model.state_dict(), path)
    epoch_end = time.time()
    print(f"{epoch+1} train time : {(epoch_end - epoch_start)}s")

# 전체 학습 종료 리포트
total_end = time.time()
print("\n" + "="*30)
print(f" 학습 완료 리포트")
print(f"="*30)
print(f" - 최고(Best) F1-score : {best_f1:.4f}")
print(f" - 전체 평균 F1-score  : {sum(f1_list)/len(f1_list):.4f}")
print(f" - 총 소요 시간        : {(total_end - total_start):.2f}s")