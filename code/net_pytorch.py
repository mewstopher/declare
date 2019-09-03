import torch.nn.functional as F

class DeclareNet(nn.module):
    def __init__(self, emb_matrix):
        super(DeclareNet, self).__init__()
        embed_art_size = emb_matrix.shape[1]
        self.embedding_art = nn.Embedding(max_features, embed_art_size)
        self.embedding_art_weight = nn.Parameter(torch.tensor(embedding_mat, dtype=torch.float32))
        self.embedding_art_weight.requires_grad = False

        self.embedding_claim = nn.Embedding(max_features, embed_size)
        self.embedding_claim_weight = nn.Parameter(torch.tensor(embedding_mat, dtype=float32))
        self.embedding_claim_weight.requires_grad = False
        self.attn_dense = nn.Linear(128, 128)
        self.lstm1 = nn.LSTM(128, 128, bidirecitonal=True)
        self.final_dense = nn.Linear(128, 32)
        self.dense_out = nn.liner(32, 1)

    def forward(self, claim_word, art_word, art_src):
        claim_wrd_emb = self.embedding(claim_word)
        mean_claim_word_emb = torch.mean(claim_wrd_emb).repeat(100)

        art_wrd_emb = self.embedding_art(art_wrd)
        clm_art_cat = torch.cat([mean_claim_word_emb, art_wrd_emb])
        attn_weights = F.tanh(self.attn_dense(clm_art_cat))
        attn_weights = F.softmax(attn_weights)

        lstm_out, _ = self.lstm1(self.embedding_art)
        lstm_max_pool = torch.max(lstm_out, 1)

        inner_pdt = torch.dot(attn_weights, lstm_max_pool)
        avg = torch.mean(inner_pdt, axis=-1).repeat(1)[:,0,:]
        final_feat = torch.cat([clm_src, avg, art_src])
        final_dense_out = self.final_dense(final_out)
        final_out = self.dense_out(final_dense_out)
        return final_out


