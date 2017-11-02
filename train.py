import sys
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform
from model import FiLM

batch_size = 64
n_epoch = 80

def train(epoch):
    train_set = DataLoader(CLEVR(sys.argv[1], transform=transform),
                    batch_size=batch_size, num_workers=4,
                    collate_fn=collate_data)

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for image, question, q_len, answer, _ in pbar:
        image, question, answer = \
            Variable(image).cuda(), Variable(question).cuda(), \
            Variable(answer).cuda()

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.data.cpu().numpy().argmax(1) \
                == answer.data.cpu().numpy()
        correct = correct.sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description('Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'. \
                            format(epoch + 1, loss.data[0],
                                moving_loss))

def valid(epoch):
    valid_set = DataLoader(CLEVR(sys.argv[1], 'val', transform=transform),
                    batch_size=batch_size, num_workers=4,
                    collate_fn=collate_data)
    dataset = iter(valid_set)

    net.train(False)
    family_correct = Counter()
    family_total = Counter()
    for image, question, q_len, answer, family in tqdm(dataset):
        image, question = \
            Variable(image).cuda(), Variable(question).cuda()

        output = net(image, question, q_len)
        correct = output.data.cpu().numpy().argmax(1) == answer.numpy()
        for c, fam in zip(correct, family):
            if c: family_correct[fam] += 1
            family_total[fam] += 1

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        for k, v in family_total.items():
            w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    print('Avg Acc: {:.5f}'.format(sum(family_correct.values()) / \
                                sum(family_total.values())))

if __name__ == '__main__':
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = FiLM(n_words).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in range(n_epoch):
        train(epoch)
        valid(epoch)

        with open('checkpoint/checkpoint_{}.model' \
            .format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net, f)
