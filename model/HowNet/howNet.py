import json
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append('../')
class HowNetLexicon:
    def __init__(self, path='../data/SingleSentenceClassification/hownet.json'):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hownet_path = os.path.abspath(os.path.join(
            project_root,"HowNet","hownet.json"
        ))
        # expect: { "词": { "synonyms": [...], "antonyms": [...] }, ... }
        with open(hownet_path, 'r', encoding='utf8') as f:
            self.lex = json.load(f)

    def get_synonyms(self, token):
        return self.lex.get(token, {}).get("synonyms", [])

    def get_antonyms(self, token):
        return self.lex.get(token, {}).get("antonyms", [])


def compute_semantic_loss(input_ids, hidden_states, vocab, lexicon, lamda=0.1, margin=1.0):
    """
     lexicon: HowNetLexicon instance
    returns: scalar tensor semantic_loss
    """
    seq_len, batch_size, H = hidden_states.shape
    loss_syn, loss_ant = 0.0, 0.0
    count_syn, count_ant = 0, 0
    if loss_ant==0:return 0
    # transpose for easier indexing: [batch_size, seq_len, H]
    hs = hidden_states.transpose(0,1)
    for b in range(seq_len):
        tokens = [vocab.itos[i] for i in input_ids[:, b].tolist()]
        for i, tok in enumerate(tokens):
            syns = lexicon.get_synonyms(tok)
            ants = lexicon.get_antonyms(tok)
            if syns:
                # find first synonym in this sentence
                for j, tok2 in enumerate(tokens):
                    if tok2 in syns:
                        v1, v2 = hs[b,i], hs[b,j]  # [H]
                        loss_syn += F.mse_loss(v1, v2)
                        count_syn += 1
                        break
            if ants:
                for j, tok2 in enumerate(tokens):
                    if tok2 in ants:
                        v1, v2 = hs[b,i], hs[b,j]
                        # margin loss: max(0, margin – ||v1–v2||)
                        dist = F.pairwise_distance(v1.unsqueeze(0), v2.unsqueeze(0))
                        loss_ant += torch.clamp(margin - dist, min=0.0).mean()
                        count_ant += 1
                        break

    if count_syn > 0:
        loss_syn = loss_syn / count_syn
    if count_ant > 0:
        loss_ant = loss_ant / count_ant

    return lamda * (loss_syn + loss_ant)