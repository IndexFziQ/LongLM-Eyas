# LongLM-Eyas

## IIE-NLP-Eyas@COSG Implement Details

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

convert each event into one line with ‘/t’ spliting

`python event2data.py`

Thus, we get result4rank_of_val/test.txt

#### Step 6:

perform ranking

`bash ranking.sh`

Thus, we get result4ranked_of_val/test.txt

#### Step 7:

del repeative words

`python data4lcs.py`

`python lrc.py`

Thus, we get result4lcs.txt, final_result.txt

#### Step 8:

`python source2jsonl.py`

Finally, we get the submission.jsonl
 
## Acknowledgement

Thanks for the baseline model [LongLM](https://github.com/thu-coai/LOT-Benchmark).


