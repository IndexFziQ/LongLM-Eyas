# NLG任务二：基于大纲的条件故事生成

### 数据样例

训练集和验证集在`./data`文件夹下，参赛选手可自行重新划分数据集

```
{
	"story":
		"有个人把神像放在驴子背上，赶着进城。凡是遇见他们的人都对着神像顶礼膜拜。驴子以为人们是向它致敬，便洋洋得意，大喊大叫，再也不肯往前走了。结果挨了驴夫狠狠的一棍。", 
	"outline":
  	["对着神像顶礼膜拜", "再也不肯往前走", "神像放在驴子", "赶着进城", "驴夫狠狠", "洋洋得意", "大喊大叫", "遇见"]
}
```

- "outline"（`list of str`）：输入的故事大纲（一个无序的短语集合）
- "story" (`str`)：目标故事

### 评价

预测结果应该有和 `test.jsonl`一样的数据格式，执行下列命令进行评价：

```shell
python eval.py prediction_file test.jsonl
```

脚本 `eval.py`的输出如下：

```python
{'bleu-1': '_', 'bleu-2': '_', 'bleu-3': '_', 'bleu-4': '_', 'distinct-1': '_', 'distinct-2': '_',  'distinct-3': '_', 'distinct-4': '_', 'coverage': '_', 'order': '_'}
```

依赖: rouge\=\=1.0.0, jieba=0.42.1, nltk=3.6.2, numpy=1.20.3

### 生成结果

在`./gen`文件夹下，`small_val.jsonl`、`base_val.jsonl`、`large_val.jsonl`分别表示`LongLM-small`、`LongLM-base`、`LongLM-large`在验证集上的生成结果。
