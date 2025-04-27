import json
import OpenHowNet

# 初始化 HowNet
hownet = OpenHowNet.HowNetDict()

# 假设你的词表文件是 vocab.txt，一行一个词
with open('vocab.txt', encoding='utf-8') as f:
    vocab = [w.strip() for w in f if w.strip()]

lexicon = {}
for word in vocab:
    # 获取同义词列表
    syns = hownet.get_synonyms(word)
    # 获取反义词列表
    ants = hownet.get_antonyms(word)
    # 过滤空结果，并去重
    syns = list({s for s in syns if s != word})
    ants = list({a for a in ants if a != word})

    if syns or ants:
        lexicon[word] = {
            'synonyms': syns,
            'antonyms': ants
        }

# 将结果写入 JSON
with open('hownet_lexicon.json', 'w', encoding='utf-8') as out:
    json.dump(lexicon, out, ensure_ascii=False, indent=2)