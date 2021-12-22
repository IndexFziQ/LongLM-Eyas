#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : split_kw_sent.py
@Author : Yuqiang Xie
@Date   : 2021/11/1
@E-Mail : indexfziq@gmail.com
"""
import json
import jsonlines
import re

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

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


def find_x(outline, sents):
    start = [0]
    end = []
    for sub_outline in outline:
        for i in range (len (sents)):
            if sub_outline in sents[i]:
                start.append (i)
                if i>0:
                    end.append(i)
            end.append (len (sents))
    outline_list = sorted (list(set(start)))
    outline_list_ = sorted (list(set(end)))

    story_spilt = []
    for j in range (len(outline_list)):
        sents_ = sents[outline_list[j]:outline_list_[j]]
        story_spilt.append (sents_)

    return story_spilt

# file_path = '/home/yuqiang.xyq/LongLM/datasets/chn/data/re_ranking'
file_path = '/home/yuqiang.xyq/LongLM/datasets/chn/data/final/test'
# set source: train/valid/test.jsonl
cases = read_jsonl(file_path+'/semi_test.jsonl')
# outputs = read_txt(file_path+'/result_small_aug.txt')
with open(file_path+'/semi_test_split.jsonl','w', encoding='utf-8') as final:
    new_cases = []
    for i in range(len(cases)):

        sents = cut_sent(cases[i]['story'])
        outline = cases[i]['outline']

        story_split = find_x(outline,sents)

        for j in range(len(outline)):
            for story in story_split:
                if outline[j] in ''.join (story):
                    final.write (json.dumps ({'story':'当前情节：' + outline[j]
                                                      + '。生成的故事：' + ''.join (story),
                                              'outline': outline}, ensure_ascii=False) + '\n')
                    break