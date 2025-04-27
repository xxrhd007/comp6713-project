import os, sys
ROOT = os.path.dirname(__file__)     # 假设 demo.py 就在 project_root 下
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import torch
import gradio as gr
from transformers import BertTokenizer,pipeline
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from model.DownstreamTasks.BertForSentenceClassification import BertForSentenceClassification
from utils.data_helpers import LoadSingleSentenceClassificationDataset
_mapping_txt = """\
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
"""
_category_map = []
for line in _mapping_txt.splitlines():
    code, zh1, zh2, en = line.strip().split()
    _category_map.append({
        "code": code,
        "zh"  : zh1 + zh2,
        "en"  : en
    })

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

def predict_and_translate(sentence: str):
    config = ModelConfig()
    device = config.device

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    loader = LoadSingleSentenceClassificationDataset(
        vocab_path=config.vocab_path,
        tokenizer=tokenizer.tokenize,
        batch_size=1,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=False
    )

    tokens = tokenizer.tokenize(sentence)
    ids = [loader.CLS_IDX] + [
        loader.vocab.stoi.get(t, loader.vocab.stoi[loader.vocab.UNK])
        for t in tokens
    ] + [loader.SEP_IDX]
    max_len = loader.max_position_embeddings - 1
    if len(ids) > max_len + 1:
        ids = ids[:max_len] + [loader.SEP_IDX]

    tensor = torch.tensor(ids, dtype=torch.long)
    batch_sent, _ = loader.generate_batch([(tensor, 0)])
    batch_sent = batch_sent.to(device)
    attention_mask = (batch_sent == loader.PAD_IDX).transpose(0, 1)
    model = BertForSentenceClassification(config, config.pretrained_model_dir)
    state_dict = torch.load(
        os.path.join(config.model_save_dir, 'model.pt'),
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    with torch.no_grad():
        logits = model(batch_sent, attention_mask=attention_mask)
        idx = logits.argmax(dim=1).item()
    item = _category_map[idx]
    class_result = f"{item['zh']} （{item['en']}，code={item['code']}）"

    translation = translator(sentence, max_length=256)[0]["translation_text"]

    return class_result, translation

demo = gr.Interface(
    fn=predict_and_translate,
    inputs=gr.Textbox(lines=2, placeholder="place a title here……"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Translation")
    ],
    title="News title classification",
    description="Input a sentence in Chinese, output: 1) news classification labels; 2) machine-translated English text."
)

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0", share=True)