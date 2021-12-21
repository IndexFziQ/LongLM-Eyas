#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from new_seq2seq_trainer import Seq2SeqTrainer
from seq2seq_training_args import Seq2SeqTrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)
from resize_embed import resize_token_embeddings
import json
import argparse
import sys
import numpy as np
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
from rouge import Rouge
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

def bleu(data):
    """
    compute rouge score
    Args:
        data (list of dict including reference and candidate):
    Returns:
            res (dict of list of scores): rouge score
    """

    res = {}
    for i in range(1, 5):
        res["bleu-%d" % i] = []

    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        for i in range(1, 5):
            res["bleu-%d" % i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference],
                                                    hypothesis=origin_candidate.strip().split(),
                                                    weights=tuple([1. / i for j in range(i)])))

    for key in res:
        res[key] = np.mean(res[key])

    return res


def repetition_distinct(eval_data):
    result = {}
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, tmp_data in enumerate(eval_data):
            ngs = ["_".join(c) for c in ngrams(tmp_data["candidate"], i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d" % i] = len(all_ngram) / float(all_ngram_num)
    return result


def rouge(ipt, cand):
    rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
    item_name = ["f", "p", "r"]

    res = {}
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s" % (name1, name2)] = []
    for k, (tmp_ipt, tmp_cand) in enumerate(zip(ipt, cand)):
        for tmp_ref in tmp_ipt.split("#"):
            # print(tmp_ref.strip())
            # print(" ".join(tmp_cand))

            # tmp_ref = tmp_ref.strip()
            # tmp_hyp = " ".join(tmp_cand).strip()

            tmp_ref = " ".join([w for w in "".join(tmp_ref.strip().split())])
            tmp_hyp = " ".join([w for w in "".join(tmp_cand.strip().split())])
            # print(tmp_ref)
            # print(tmp_hyp)
            try:
                tmp_res = Rouge().get_scores(refs=tmp_ref, hyps=tmp_hyp)[0]
                for name1 in rouge_name:
                    for name2 in item_name:
                        res["%s-%s" % (name1, name2)].append(tmp_res[name1][name2])
            except:
                continue
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s" % (name1, name2)] = np.mean(res["%s-%s" % (name1, name2)])
    return {"coverage": res["rouge-l-r"]}


def LCS(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def Recon_LCS(x, y, exclusive=True):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = LCS(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    if len(recon_list):
        return "".join(recon_list).strip()
    else:
        return ""
    # return Ngrams(recon_list, exclusive=exclusive)
    # return recon_tuple


def lcs3_dp(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * (len(input_y) + 1)) for i in range(len(input_x) + 1)]
    maxlen = maxindex = 0
    for i in range(1, len(input_x) + 1):
        for j in range(1, len(input_y) + 1):
            if i == 0 or j == 0:  # 在边界上，自行+1
                dp[i][j] = 0
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > maxlen:  # 随时更新最长长度和长度开始的位置
                    maxlen = dp[i][j]
                    maxindex = i - maxlen
                    # print('最长公共子串的长度是:%s' % maxlen)
                    # print('最长公共子串是:%s' % input_x[maxindex:maxindex + maxlen])
            else:
                dp[i][j] = 0
    # for dp_line in dp:
    #     print(dp_line)
    return input_x[maxindex:maxindex + maxlen]


def inversenum(a):
    num = 0
    all_num = 0
    for i in range(0, len(a)):
        for j in range(i, len(a)):
            if a[i] > a[j]:
                num += 1
            all_num += 1
    return num / float(all_num)


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1


def order(ipt, cand, kw2id):
    num = []
    for k, (tmp_ipt, tmp_cand, tmp_kw2id) in enumerate(zip(ipt, cand, kw2id)):
        # all_pos = [[]]
        pos = []
        kw_list = list(tmp_kw2id.keys())
        kw_list.reverse()

        for tmp_ref in kw_list:
            tmp_ref = "".join(tmp_ref.strip().split())
            tmp_hyp = "".join(tmp_cand.strip().split())
            lcs = lcs3_dp(tmp_ref, tmp_hyp)
            if len(lcs) > 1:
                pos.append(tmp_hyp.find(lcs))
            else:
                pos.append(-1)
        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])

        new_rank = [-1 for _ in idlist]
        for idl, ord in zip(idlist, orderlist):
            new_rank[idl] = tmp_kw2id[kw_list[ord]]
        num.append(1 - inversenum(new_rank))

    return {"order": np.mean(num)}


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()
    return data


def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])


def softmax(score):
    softmax_x = []
    for i_score in score:
        softmax_x.append(i_score / np.sum(score))

    return softmax_x


def overall_score_w(human, small):
    w = []
    for i in range(6):
        w_ = human[i] / small[i]
        w.append(w_)

    return softmax(w)


def compute(golden_file, pred_file, return_dict=True):
    golden_data = load_file(golden_file)
    pred_data = load_file(pred_file)

    if len(golden_data) != len(pred_data):
        raise RuntimeError("Wrong Predictions")

    ipt = ["#".join(g["outline"]) for g in golden_data]
    truth = [g["story"] for g in golden_data]
    pred = [p["story"] for p in pred_data]

    kw2id = []
    for i1, t1 in zip(ipt, truth):
        kw_list = i1.strip().split("#")
        pos = [t1.strip().find(kw.strip()) for kw in kw_list]

        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])
        kw2id.append({})
        for idl, ord in zip(idlist, orderlist):
            kw2id[-1][kw_list[ord]] = idl

    eval_data = [{"reference": proline(g["story"]), "candidate": proline(p["story"])} for g, p in
                 zip(golden_data, pred_data)]
    res = bleu(eval_data)
    bleu_score = bleu(eval_data)
    res.update(repetition_distinct(eval_data))
    dist_score = repetition_distinct(eval_data)
    res.update(rouge(ipt=ipt, cand=pred))
    cover_score = rouge(ipt=ipt, cand=pred)
    res.update(order(ipt=ipt, cand=pred, kw2id=kw2id))
    order_score = order(ipt=ipt, cand=pred, kw2id=kw2id)

    # test-w
    # human = [100, 100, 21.28, 39.54, 100, 100]
    # small = [29.42, 17.97, 16.17, 29.15, 83.87, 64.10]
    # val-w
    human = [100, 100, 23.47, 42.17, 100, 100]
    small = [26.58, 16.04, 17.90, 31.38, 83.64, 63.15]
    w = overall_score_w(human, small)

    overall_score = w[0] * bleu_score['bleu-1'] + w[1] * bleu_score['bleu-2'] + w[2] * dist_score['distinct-3'] + w[3] * \
                    dist_score['distinct-4'] + w[4] * cover_score['coverage'] + w[5] * order_score['order']

    # for key in res:
    #     res[key] = "_"
    return res, overall_score


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=100, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=100, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    train_name: Optional[str] = field(default="train", metadata={"help": "# data name for training."})


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    except:
        print("*"*20)
        print("train from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config=config)
        print("*"*20)


    num_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param += torch.numel(param)
    print("="*10)
    print("# Parameters:", num_param)
    # model.resize_token_embeddings(len(tokenizer))
    # resize_token_embeddings(model, len(tokenizer))
    vocab_size = len(tokenizer)
    print("vocab_size:", vocab_size)
    print("="*10)

    # use task specific params
    use_task_specific_params(model, data_args.task)
    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang)
    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    dataset_class = Seq2SeqDataset

    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path=data_args.train_name,
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else None
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(
            tokenizer, data_args, model.config.decoder_start_token_id, training_args.tpu_num_cores
        ),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )
    # trainer.save_model()  # this also saves the tokenizer
    # exit()

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics(data_args.train_name, metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        test_output = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test
        test_preds = tokenizer.batch_decode(
            test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        test_preds = lmap(str.strip, test_preds)
        write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))
        # set source: train/valid/test.jsonl
        cases = read_jsonl(os.path.join("/home/yuqiang.xyq/LongLM/datasets/chn/data/tmp/val.jsonl"))
        outputs = read_txt(os.path.join(training_args.output_dir, "test_generations.txt"))

        # set target: train/val/test.source or target
        with open("/home/yuqiang.xyq/LongLM/datasets/chn/data/tmp/pred_val.jsonl", 'w', encoding='utf-8') as final:
            for i in range(len(cases)):
                # outline = ' '.join (cases[i]['outline'])
                outline = ''
                cases[i]['story'] = outputs[i] + outline
                final.write(json.dumps(cases[i], ensure_ascii=False) + '\n')
        all_score, overall_score = compute("/home/yuqiang.xyq/LongLM/datasets/chn/data/tmp/val.jsonl"
                                           ,"/home/yuqiang.xyq/LongLM/datasets/chn/data/tmp/pred_val.jsonl")
        metrics = trainer.evaluate(metric_key_prefix="val", overall_score=overall_score)
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

                all_metrics.update(metrics)

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
