# LongLM-Eyas

**IIE-NLP-Eyas@OutGen: Chinese Outline-guided Story Generation via a Progressive Plot-Event-Story Framework** [[PDF]()]

**Team Members:** *Yuqiang Xie, Yunpeng Li, Wei Peng, Ping Guo and Luxi Xing.* 

**Org.:** Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China.

**Codes are contributed by:**
- Yuqiang Xie (Baselines, Data Pre-processing, Event Generation)
- Yunpeng Li (Event Ranking)
- Wei Peng (LCS).

## Plot-Event-Story (PES)

**Outline:**
- Plot (Step 1-2)
- Event (Step 3-4)
- Story (Step 5-8)

### A Simple Guide:

#### Step 1:

convert train/val/test.jsonl into events of each plot

`python ./tools/split_kw_sent.py`

-> train/val/test_split.jsonl

#### Step 2:

convert train/val/test_split.jsonl into bart format (source and target)

`python ./tools/convert_bartio.py`

-> train/val/test.source/target

#### Step 3:

Train/Eval/Test using LongLM-small model

`bash ./longlm/finetune_deepspeed_iie.sh`

The best model will be in ./save_model

#### Step 4:

Generating stories by Top-p sampling:

`python ./baselines/generation/gen.py`

-> result_of_val/test.txt

#### Step 5:

convert each event into one line with ‘/t’ splitting

`python ./tools/event2data.py`

-> result4rank_of_val/test.txt

#### Step 6:

perform ranking

`python ./tools/outline_reranking.py`

-> train/val_reranking.jsonl

`python ./tools/process_nsp_data.py`

-> train/val_nsp.txt

`python ./tools/story_nsp.py`

-> rerank_test.txt


#### Step 7:

del repetitive words

`python ./tools/data4lcs.py`

-> result4lcs.txt

`python ./tools/lrc.py`

-> final_result.txt


#### Step 8:

`python ./tools/source2jsonl.py`

-> submission.jsonl


### Parameters for Baselines and Event Generation:

```
learning rate: 3e-5
epoch: {5, 10}
top-k: K=40
K's temperature: 0.9
batch size: 8
```
 
## Acknowledgement

Thanks for the baseline model [LongLM](https://github.com/thu-coai/LOT-Benchmark).


