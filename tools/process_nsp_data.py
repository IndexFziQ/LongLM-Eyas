import jsonlines
import random

# process the jsonl file for nsp
def process(spilt_file, rerank_file, output_file):
    f_w = open(output_file, 'w')
    temp_outline = []
    story = []
    outlines = []
    persentence_outline = []
    sentences = []
    temp_outline1 = []
    with jsonlines.open(spilt_file) as f:
        for item in f:
            if(item['outline'] != temp_outline):
                if(sentences != []):
                    story.append(sentences)
                    outlines.append(persentence_outline)
                sentences = []
                temp_outline = item['outline']
                temp_outline1 = [word for word in temp_outline]
                persentence_outline = []
            index = item['story'][5:].index('：')
            if(item['story'][5:][index + 1:] not in sentences):
                persentence_outline.append(temp_outline1[len(sentences)])
                sentences.append(item['story'][5:][index + 1:])
            else:
                temp_outline1.pop(len(sentences))
    write_lines = []
    with jsonlines.open(rerank_file) as f:
        for item, sentences, persentence_outline in zip(f, story, outlines):
            temp = []
            for sentence, word in zip(sentences, persentence_outline):
                index = item['outline'].index(word)
                temp.append([sentence, index])
            temp = sorted(temp, key=lambda x: x[1])
            for i in range(len(temp)):
                temp[i][1] = str(i)
            write_lines.append(temp)
    write_list = []
    story = []
    for perstory in write_lines:
        for line in perstory:

            if (line[1][0] == '0' and story != []):
                if (len(story) == 1):
                    continue
                for i in range(len(story)):
                    if (i < len(story) - 1):
                        sentence1 = story[i]
                        sentence2 = story[i + 1]
                        write_list.append('\t'.join([sentence1, sentence2]) + '\t' + str(0) + '\n')
                    if (i != 0 or len(story) != 2):
                        rand_int = random.randint(0, len(story) - 1)
                        while (rand_int in [i, i + 1]):
                            rand_int = random.randint(0, len(story) - 1)
                        sentence1 = story[i]
                        sentence2 = story[rand_int]
                        write_list.append('\t'.join([sentence1, sentence2]) + '\t' + str(1) + '\n')
                story = []
                story.append(line[0])
            else:
                story.append(line[0])

    for line in write_list:
        f_w.write(str(line))

if __name__ == '__main__':
    data_split = 'train' # train/val
    split_file = data_split + '_split.jsonl'
    rerank_file = data_split + '_reranking.jsonl'
    output_file = data_split + '_nsp.txt'
    process(split_file, rerank_file, output_file)