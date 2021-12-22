#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : source2jsonl.py
@Author : Yuqiang Xie
@Date   : 2021/12/20
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

file_path = './data/test'
# set source: train/valid/test.jsonl
cases = read_jsonl(file_path+'/test.jsonl')
outputs = read_txt(file_path+'/result.txt')

# set target: train/val/test.source or target
with open(file_path+'/iie_test.jsonl','w', encoding='utf-8') as final:
    for i in range(len(cases)):
        cases[i]['story'] = outputs[i]
        final.write (json.dumps (cases[i], ensure_ascii=False) + '\n')
