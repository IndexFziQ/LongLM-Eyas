#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : find_first_sentence.py
@Author : Yuqiang Xie
@Date   : 2021/12/20
@E-Mail : indexfziq@gmail.com
"""
# compute overlap bewtween each sentences of val data,
# with key words set of first senntences in train data

import json
import jsonlines
import re
import jieba

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()

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

def proline(line):

    return [w for w in jieba.cut("".join(line))]


def find_x(outline, sents):
    start = [0]
    end = []
    for sub_outline in outline:
        for i in range(len(sents)):
            if sub_outline in sents[i]:
                start.append (i)
                if i>0:
                    end.append(i)
            end.append(len(sents))
    outline_list = sorted(list(set(start)))
    outline_list_ = sorted(list(set(end)))

    story_spilt = []
    for j in range(len(outline_list)):
        sents_ = sents[outline_list[j]:outline_list_[j]]
        story_spilt.append (sents_)

    return story_spilt

def diff(listA, listB):
    retB = list(set(listA).intersection(set(listB)))

    return retB

file_path = './data'

cases = read_jsonl(file_path+'/train.jsonl')
gens = read_jsonl(file_path+'/val.jsonl')

first_sentence_elements = []
for i in range(len(cases)):
    sents = cut_sent(cases[i]['story'])
    outline = cases[i]['outline']

    story_split = find_x(outline, sents)
    tmp = proline(story_split[0])
    first_sentence_elements = tmp + first_sentence_elements
    first_sentence_elements = list(set(first_sentence_elements))

count = [0 for i in range (8)]

for i in range(len(gens)):
    sents = cut_sent(gens[i]['story'])
    outline = gens[i]['outline']

    story_split = find_x(outline, sents)
    position = []
    for story in story_split:
        tmp = proline(story)
        coverage = diff(tmp,first_sentence_elements)
        position.append(len(coverage))

    max_list = max (position)
    max_index = position.index (max (position))

    pos = [i for i in range(8)]

    for i in pos:
        if max_index == i:
            count[i] = count[i] +1

print (count)



