from transformers import AdamW,BertConfig
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForNextSentencePrediction
import transformers
transformers.logging.set_verbosity_error()
import torch.nn as nn
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def inversenum(a):
    num = 0
    all_num = 0
    for i in range(0,len(a)):
        for j in range(i,len(a)):
            if a[i] > a[j]:
                num += 1
            all_num += 1
    return num / float(all_num)

class bertconfig():

    def __init__(self):
        self.seqlength = 512
        self.train_batchsize = 8
        self.test_batchsize = 8
        self.evalbatchsize = 8
        self.hidden_dropout_prob = 0.1
        self.num_labels = 2
        self.learning_rate = 3e-5
        self.weight_decay = 1e-2
        self.epochs = 15

class storydata(Dataset):
    def __init__(self,datadir,tokenizer,config):

        text = datadir
        f = open(text)
        lines = f.readlines()
        self.input_ids, self.token_type, self.labels, self.attention_masks = tokenize(lines, tokenizer, config)
    def __getitem__(self, index):
        return self.input_ids[index], self.token_type[index], self.labels[index], self.attention_masks[index]

    def __len__(self):
        return len(self.input_ids)


def tokenize(lines,tokenizer,config):

    input_ids = []
    token_types = []
    labels = []
    attention_masks = []
    pad_id = int(tokenizer.convert_tokens_to_ids("[PAD]"))
    dict = {'0': 0, '1': 1}
    for i in range(0,len(lines)):
        label = []
        sentence1, sentence2, temp_label = lines[i].split('\t')
        temp_label = temp_label[0]
        label.append(dict[temp_label])
        input = tokenizer.encode_plus(sentence1, sentence2, max_length=512)
        temp_id = input['input_ids']
        attention_mask = input['attention_mask']
        token_type = input['token_type_ids']
        if(len(temp_id) > 512):
            temp_id = temp_id[:512]
            token_type = token_type[:512]
            attention_mask = attention_mask[:512]
        while(len(temp_id)<512):
            temp_id.append(pad_id)
            attention_mask.append(0)
            token_type.append(0)
        input_ids.append(temp_id)
        labels.append(label)
        token_types.append(token_type)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(token_types), np.array(labels), np.array(attention_masks)

def train(epochs, model, train_loader, optimizer, criterion, device,tokenizer,config):

    print('Begin training！')
    best = 0.0
    train_losses = []
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            input_ids, token_type_ids, labels, attention_masks = data[0], data[1], data[2], data[3]
            labels = labels.unsqueeze(1)
            input_ids, token_type_ids, labels, attention_masks = input_ids.long(), token_type_ids.long(), \
                                                                 labels.long(), attention_masks.long()
            optimizer.zero_grad()
            input_ids, token_type_ids, labels, attention_masks = input_ids.to(device), token_type_ids.to(device), \
                                                                 labels.to(device), attention_masks.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels,
                           attention_mask=attention_masks)
            loss = output[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 50 == 0:
                print("epoch: ", epoch, "\t","step: ", i, "\t", "current loss:", epoch_loss)
        print("epoch: ", epoch, "\t", "current loss:", epoch_loss )
        train_losses.append(epoch_loss)
        mean = evaluate(model)

        if(mean > best):
            best = mean
            print('best model saved!')
            torch.save(model.state_dict(), 'bert.pth')

def evaluate(model):
    import time
    softmax = nn.Softmax()
    print(time.clock())
    f = open('val_nsp.txt')
    lines = f.readlines()
    story = []
    acc = []
    for line in lines:
        if (line.split('\t')[1][0] == '0' and story != []):
            nsp = np.zeros([len(story), len(story)])
            # print(story[0])
            for i in range(len(story)):
                for j in range(len(story)):
                    if (j != i):
                        sentence1 = story[i]
                        sentence2 = story[j]
                        input = tokenizer.encode_plus(sentence1, sentence2, max_length=512)
                        input_ids = torch.Tensor(input['input_ids']).cuda().long().view(1, -1)
                        attention_masks = torch.Tensor(input['attention_mask']).cuda().long().view(1, -1)
                        token_type_ids = torch.Tensor(input['token_type_ids']).cuda().long().view(1, -1)
                        # print(input_ids)
                        output = model(input_ids=input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids)
                        nsp[i][j] = output.logits[0][0].item()
            order = [0]
            for i in range(len(story) - 1):
                index = order[-1]
                for j in range(len(story)):
                    nsp[j][index] = 0.0
                temp_nsp = nsp[index]
                new_index = np.argmax(temp_nsp)
                order.append(new_index)
            acc.append(1.0 - inversenum(order))
            story = []
            story.append(line.split('\t')[0])
        else:
            story.append(line.split('\t')[0])

    print('eval: ', np.array(acc).mean())
    return np.array(acc).mean()


def inference(model, input_file, output_file):
    f = open(input_file)
    f_w = open(output_file, 'w')
    lines = f.readlines()
    for line in lines:
            story = line.split('\t')[:-1]
            nsp = np.zeros([len(story), len(story)])
            for i in range(len(story)):
                for j in range(len(story)):
                    if (j != i):
                        sentence1 = story[i]
                        sentence2 = story[j]
                        input = tokenizer.encode_plus(sentence1, sentence2, max_length=512)
                        input_ids = torch.Tensor(input['input_ids']).cuda().long().view(1, -1)
                        attention_masks = torch.Tensor(input['attention_mask']).cuda().long().view(1, -1)
                        token_type_ids = torch.Tensor(input['token_type_ids']).cuda().long().view(1, -1)
                        output = model(input_ids=input_ids, attention_mask=attention_masks,
                                       token_type_ids=token_type_ids)
                        nsp[i][j] = output.logits[0][0].item()
            new = np.sum(nsp, axis=0)
            order = [np.argmin(new)]
            for i in range(len(story) - 1):
                index = order[-1]
                for j in range(len(story)):
                    nsp[j][index] = 0.0
                temp_nsp = nsp[index]
                new_index = np.argmax(temp_nsp)
                order.append(new_index)
            write_line = [story[ind] for ind in order]
            f_w.write('\t'.join(write_line) + '\n')

    return 1

if __name__ == '__main__':

    """
        setting the pretrained_model_path and training flag
    """
    pretrained_model_path = ''
    is_training = True

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path, do_lower_case=True)
    config = bertconfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bertconfig = BertConfig.from_pretrained(pretrained_model_path + "config.json",
                                            num_labels=2, hidden_dropout_prob=config.hidden_dropout_prob)
    model = BertForNextSentencePrediction.from_pretrained(pretrained_model_path + "pytorch_model.bin", config=bertconfig)
    mode = model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if(is_training):
        # train_file: the file for finetuning bert nsp. format: sentence1 + '\t' + sentence2 + '\t' + label
        train_file = 'train_nsp.txt'
        data = storydata(train_file, tokenizer, config)
        train_loader = DataLoader(data, batch_size=config.train_batchsize, shuffle=True)
        train(config.epochs, model, train_loader, optimizer, criterion, device, tokenizer, config)
    else:
        inference_file_path = 'result4rank_of_val/test.txt'
        output_file_path = 'result4rank_of_val/rerank_test.txt'
        model_save_path = 'bert.pth'
        if(os.path.exists(model_save_path)):
            print('Existing finetuned nsp model, load it!')
            model.load_state_dict(torch.load(model_save_path))
        else:
            print('Do not existing finetuned nsp model, using the original bert model for nsp!')
        inference(model, inference_file_path, output_file_path)