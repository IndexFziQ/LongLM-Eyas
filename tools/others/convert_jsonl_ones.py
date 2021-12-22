#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : convert_jsonl_ones.py
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

file_path = './data'
# set source: train/valid/test.jsonl
cases = read_jsonl(file_path+'/val.jsonl')


len_outlines = [0]
for i in range (len (cases)):
    len_outline = len(cases[i]['outline'])
    len_outlines.append(len_outline)
import numpy as np

lens = np.cumsum(len_outlines)
start=lens[:-1]
end=lens[1:]

outputs = read_txt(file_path+'/data4rank.txt')

# set target: train/val/test.source or target
with open(file_path+'/final_result.jsonl','w', encoding='utf-8') as final:
    for i in range(len(cases)):
        cases[i]['story'] = ''.join(outputs[start[i]:end[i]])
        final.write (json.dumps (cases[i], ensure_ascii=False) + '\n')
