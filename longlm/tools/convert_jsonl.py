#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : convert_bartio.py
@Author : Yuqiang Xie
@Date   : 2021/11/1
@E-Mail : indexfziq@gmail.com
"""
import json
import jsonlines

def read_txt(input_file):
    "Read a text file"
    lines = []
    with open(input_file, "r") as f:
        for line in f.readlines ():
            line = line.strip('\n').strip('<extra_id_1>')
            lines.append(line)
    return lines

def read_jsonl(input_file):
    "Read a jsonl file"
    lines = []
    with open(input_file, mode='r') as json_file:
        reader = jsonlines.Reader(json_file)
        for instance in reader:
            lines.append(instance)
    return lines

# file_path = '/home/yuqiang.xyq/LongLM/datasets/chn/data/re_ranking'
file_path = '/home/yuqiang.xyq/LongLM/datasets/chn/data'
# set source: train/valid/test.jsonl
cases = read_jsonl(file_path+'/test.jsonl')
outputs = read_txt(file_path+'/deal_lcs_ep10_norank.txt')

# set target: train/val/test.source or target
with open(file_path+'/small_one_model_val_lcs_ep10_norank.jsonl','w', encoding='utf-8') as final:
    for i in range(len(cases)):
        # outline = ' '.join (cases[i]['outline'])
        outline = ''
        cases[i]['story'] = outputs[i].replace('\t','') + outline
        final.write (json.dumps (cases[i], ensure_ascii=False) + '\n')
