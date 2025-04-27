

import sys

sys.path.append("../")
from utils import logger_init
from model import BertConfig
from model import BertModel
from transformers import BertTokenizer
import os
import torch
import logging


#


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        logger_init(log_file_name='CLS', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value


if __name__ == '__main__':
    config = ModelConfig()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    bert = BertModel.from_pretrained(config, config.pretrained_model_dir)
    sentences = ["张艺兴黄金瞳片场，导演能给个合适的帽子不?","故宫如何修文物？文物医院下月向公众开放","深圳房价是沈阳6倍就是因为经济？错！"]
    encode_input = bert_tokenize(sentences, return_tensors='pt', padding=True)
    input_ids = encode_input["input_ids"].transpose(0, 1)
    token_type_ids = encode_input["token_type_ids"].transpose(0, 1)
    attention_mask = encode_input["attention_mask"] == 0
    print(attention_mask)
    with torch.no_grad():
        bert.eval()
        pooled_output, _ = bert(input_ids, attention_mask)
        print("pooled_output:\n", pooled_output)


