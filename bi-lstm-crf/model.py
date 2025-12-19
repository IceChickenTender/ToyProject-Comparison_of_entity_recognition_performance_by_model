import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)

        # 1. 임베딩 층 (Word Embedding)
        # BERT와 달리 처음부터 학습하거나, Word2Vec/FastText 등을 로드해서 쓸 수 있습니다.
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 2. Bi-LSTM 층
        # batch_first=True로 설정해야 (Batch, Seq, Feature) 순서로 처리됩니다.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # 3. Hidden Layer to Tag Space
        # LSTM의 출력(hidden_dim)을 태그 개수(target_size)로 변환
        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)

        # 4. CRF 층 (pytorch-crf 라이브러리 활용)
        self.crf = CRF(self.target_size, batch_first=True)

    def forward(self, input_ids, tags, mask):
        """
        학습(Training) 시 사용: Loss(Negative Log Likelihood) 반환
        """
        # 1. 임베딩
        embeds = self.word_embeds(input_ids)

        # 2. LSTM 통과
        lstm_out, _ = self.lstm(embeds)

        # 3. 태그 공간으로 투영 (Emissions)
        emissions = self.hidden2tag(lstm_out)

        # 4. CRF Loss 계산 (마스크 적용 필수!)
        # -log_likelihood를 반환하므로, 이를 최소화(minimize)하는 방향으로 학습하면 됩니다.
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, input_ids, mask):
        """
        추론(Inference) 시 사용: 가장 높은 확률의 태그 시퀀스 반환
        """
        embeds = self.word_embeds(input_ids)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        # Viterbi 알고리즘을 통해 가장 최적의 경로(Best Path) 추출
        best_paths = self.crf.decode(emissions, mask=mask)
        return best_paths