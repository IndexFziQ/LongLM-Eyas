#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : outline_reranking.py
@Author : Yuqiang Xie
@Date   : 2021/11/3
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

file_path = '/home/yuqiang.xyq/LongLM/datasets/chn/data/final/test'
# set source: train/valid/test.jsonl
cases = read_jsonl(file_path+'/test.jsonl')



# set target: train/val/test.source or target
with open(file_path+'/test.source','w', encoding='utf-8') as train_source,\
    open(file_path+'/test.target','w', encoding='utf-8') as train_target:
    for feature in cases:
        outline = json.dumps(feature['outline'], ensure_ascii=False)
        outline = outline.replace('[','')
        outline = outline.replace(']','')
        outline = outline.replace('"', '')
        outline_list = outline.strip().split(', ')
        story = json.dumps(feature['story'], ensure_ascii=False).replace ('"', '')
        # re-ranking
        position = {}
        for sub_outline in outline_list:
            position[sub_outline] = story.find(sub_outline)
        outline_list = sorted(position.items(), key = lambda kv:(kv[1], kv[0]))
        keys=[]
        for key in outline_list:
            keys.append(key[0])
        outline = '#'.join(keys)

        train_source.write(outline + '<extra_id_1>' + '\n')
        train_target.write('<extra_id_1>'+ story + '\n')

