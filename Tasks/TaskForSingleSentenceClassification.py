import sys

from model.HowNet import HowNetLexicon
from model.HowNet.howNet import compute_semantic_loss

sys.path.append('../')
from model import BertForSentenceClassification
from model import BertConfig
from utils import LoadSingleSentenceClassificationDataset
from utils import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time
lexicon = HowNetLexicon()
λ_sem = 0.2  # Semantic consistency weight
class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'toutiao_train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'toutiao_val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'toutiao_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.how=os.path.join(self.dataset_dir,'hownet.json')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 64
        self.max_sen_len = None
        self.num_labels = 15
        self.epochs = 10
        self.model_val_per_epoch = 2
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value



def train(config,λ_sem=0.1):
    model = BertForSentenceClassification(config,config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path,map_location=torch.device('cpu'))
        model.load_state_dict(loaded_paras)
        logging.info("## load model......")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            sem_loss = compute_semantic_loss(
                input_ids=sample,
                hidden_states=torch.zeros([512,200,768]),  # Last hidden layer [seq_len, batch_size, hidden_size]
                vocab=data_loader.vocab,
                lexicon=lexicon,
                lamda=λ_sem
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()+sem_loss*λ_sem
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)


def inference(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## load model......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    acc = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    logging.info(f"Acc on test:{acc:.3f}")


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


if __name__ == '__main__':
    model_config = ModelConfig()
    start=time.time()
    train(model_config)
    inference(model_config)
    end=time.time()
    print("Training time is :",end-start)