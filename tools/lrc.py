#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File   : lrc.py
@Author : Wei Peng, Yuqiang Xie
@Date   : 2021/12/20
"""
import numpy as np
import json
import os

def longestCommonSequence(str_one, str_two, case_sensitive=True):
    """
    str_one 和 str_two 的最长公共子序列
    :param str_one: 字符串1
    :param str_two: 字符串2（正确结果）
    :param case_sensitive: 比较时是否区分大小写，默认区分大小写
    :return: 最长公共子序列的长度
    """
    len_str1 = len(str_one)
    len_str2 = len(str_two)
    # 定义一个列表来保存最长公共子序列的长度，并初始化
    record = [[0 for i in range(len_str2 + 1)] for j in range(len_str1 + 1)]
    for i in range(len_str1):
        for j in range(len_str2):
            if str_one[i] == str_two[j]:
                record[i + 1][j + 1] = record[i][j] + 1
            elif record[i + 1][j] > record[i][j + 1]:
                record[i + 1][j + 1] = record[i + 1][j]
            else:
                record[i + 1][j + 1] = record[i][j + 1]

    return record[-1][-1]


def LCS(s1, s2):
    size1 = len(s1) + 1
    size2 = len(s2) + 1

    chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
    for i in list(range(1, size1)):
        chess[i][0][0] = s1[i - 1]
    for j in list(range(1, size2)):
        chess[0][j][0] = s2[j - 1]

    for i in list(range(1, size1)):
        for j in list(range(1, size2)):
            if s1[i - 1] == s2[j - 1]:
                chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
            elif chess[i][j - 1][1] > chess[i - 1][j][1]:
                chess[i][j] = ['←', chess[i][j - 1][1]]
            else:
                chess[i][j] = ['↑', chess[i - 1][j][1]]

    i = size1 - 1
    j = size2 - 1
    s3 = []
    while i > 0 and j > 0:
        if chess[i][j][0] == '↖':
            s3.append(chess[i][0][0])
            i -= 1
            j -= 1
        if chess[i][j][0] == '←':
            j -= 1
        if chess[i][j][0] == '↑':
            i -= 1
    s3.reverse()
    return ''.join(s3)


def getNumofCommonSubstr(str1, str2):

    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)]
    maxNum = 0
    p = 0

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i + 1

    return str1[p-maxNum:p], maxNum


if __name__ == '__main__':
    with open("./data/result4lcs.txt", 'r', encoding='utf-8') as infile:
        list_text = []
        for line in infile.readlines():
            outlines = line.strip().split('\t')[-1].split('#')
            gen = line.strip().split('\t')[:-1]
            start = gen[0]
            for i in range(len(gen)):
                for j in range(i+1, len(gen)):
                    long_str = getNumofCommonSubstr(gen[i], gen[j])[0]
                    if long_str in outlines:
                        continue
                    else:
                        gen[j] = gen[j].replace(long_str, "")

            text = "".join(gen)
            list_text.append(text)
    with open("./data/final_result_test.txt", 'w', encoding='utf-8') as outfile:
        for text in list_text:
            outfile.write(text)
            outfile.write('\n')
