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
cases = read_jsonl(file_path+'/train_split.jsonl')

# set target: train/val/test.source or target
with open(file_path+'/train.source','w', encoding='utf-8') as train_source,\
    open(file_path+'/train.target','w', encoding='utf-8') as train_target:
    for feature in cases:
        outline = json.dumps(feature['outline'], ensure_ascii=False)
        outline = outline.replace('[','')
        outline = outline.replace(']','')
        outline = outline.replace('"', '')
        outline = '#'.join(outline.strip().split(', '))
        train_source.write(outline+'<extra_id_2>'+  json.dumps(feature['story'], ensure_ascii=False).split('。生成的故事：')[0].replace ('"', '') + '<extra_id_1>' + '\n')
        train_target.write('<extra_id_1>'+ json.dumps(feature['story'], ensure_ascii=False).split('。生成的故事：')[1].replace ('"', '') + '\n')

