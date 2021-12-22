# LongLM-Eyas

IIE-NLP-Eyas@COSG: Chinese Outline-guided Story Generation via a Progressive Plot-Event-Story Framework

## Implement Details

### How to use:

#### Step 1:

convert train/val/test.jsonl into events of each plot

`python split_kw_sent.py`

Thus, we get train/val/test_split.jsonl

#### Step 2:

convert train/val/test_split.jsonl into bart format (source and target)

`python convert_bartio.py`

Thus, we get train/val/test.source/target

#### Step 3:

Train/Eval/Test using LongLM-small model

`bash finetune_deepspeed_iie.sh`

The best model will be in ./save_model

#### Step 4:

Generating stories by Top-p sampling:

`python gen.py`

Thus, we get result_of_val/test.txt

#### Step 5:

convert each event into one line with ‘/t’ splitting

`python event2data.py`

Thus, we get result4rank_of_val/test.txt

#### Step 6:

perform ranking

`python outline_reranking.py`

train/val_reranking.jsonl

`python process_nsp_data.py`

train/val_nsp.txt

`python story_nsp.py`

rerank_test.txt

Thus, we get result_ranked_of_val/test.txt

#### Step 7:

del repetitive words

`python data4lcs.py`

`python lrc.py`

Thus, we get result4lcs.txt, final_result.txt

#### Step 8:

`python source2jsonl.py`

Finally, we get the submission.jsonl

### Parameters:

```
learning rate: 3e-5
epoch: {5, 10}
top-k: K=40
K's temperature: 0.9
batch size: 8
```
 
## Acknowledgement

Thanks for the baseline model [LongLM](https://github.com/thu-coai/LOT-Benchmark).


