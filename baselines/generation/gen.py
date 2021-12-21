import traceback
import sys
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import numpy as np
from transformers import (
        T5Tokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )
import random

device = "cuda:0"
print("using %s"%device)
model_name_path = "/home/yuqiang.xyq/LongLM/longlm/longlm_small_one_model_ep10"
name = "data"
task_name = "one/val"
with open("./%s/%s.source"%(name, task_name), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/%s.target"%(name, task_name), "r") as fin:
    opt = [line.strip() for line in fin]

import sys
from unicodedata import category
chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))
def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def pro(token_list, tokenizer):
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    string = "".join(string.strip().split())
    string = strB2Q(string)
    return string

tokenizer = T5Tokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id

tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

model = T5ForConditionalGeneration.from_pretrained(model_name_path).to(device)
file_out = "./result_small_one_ep10.txt"
print("write to %s"%file_out)
# input_ids = ['了让小猴抱紧皮球#大象伯伯救起#小猴明明#一不小心掉进#狗熊刚刚#菲菲摔倒#皮球抢#球还给<extra_id_1>']
# input_ids = tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
# print(input_ids)
# gen = model.generate(input_ids, do_sample=True, max_length=512, top_k=40, temperature=0.9, decoder_start_token_id=1)
# print (gen)
from tqdm import tqdm
pbar = tqdm(total=round(len(ipt)/4+1))

with open(file_out, "w") as fout:
    batch_size = 4 #16
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            try:
                gen = model.generate (input_ids, do_sample=True, max_length=512, top_k=40, temperature=0.9,
                                      decoder_start_token_id=1)
            except RuntimeError:
                print('Runtime Error')
                continue

            for ip, op, truth in zip(ipt[st:ed], gen, opt[st:ed]):
                fout.write(pro(op, tokenizer)+"\n")
            pbar.update(1)

pbar.close()
