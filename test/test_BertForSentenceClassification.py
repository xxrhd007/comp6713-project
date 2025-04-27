import sys

sys.path.append('../')
from model import BertForSentenceClassification
from model import BertConfig
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    json_file = '../bert_base_chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    config.__dict__['num_labels'] = 15
    config.__dict__['num_hidden_layers'] = 12
    model = BertForSentenceClassification(config)

    input_ids = torch.tensor([[ 101, 2476, 5686, 1069, 7942, 7032, 4749, 4275, 1767, 8024, 2193, 4028,
         5543, 5314,  702, 1394, 6844, 4638, 2384, 2094,  679, 8043,  102,    0,
            0,    0,    0,    0,    0,    0,    0,    0]]).transpose(1,0)  # [src_len,batch_size]
    attention_mask = torch.tensor([[False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True]])  # [batch_size,src_len]
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask)
    print(logits.shape)
    writer = SummaryWriter('./runs')
    writer.add_graph(model, input_ids)
