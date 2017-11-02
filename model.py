import torch
from torch import nn
from torch.nn.init import kaiming_uniform, normal
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_blocks, filter_size):
        super().__init__()

        self.conv = nn.ModuleList()

        for i in range(n_blocks):
            if i == 0:
                n_filter = 3

            else:
                n_filter = filter_size

            self.conv.append(nn.Conv2d(n_filter, filter_size,
                                    [4, 4], 2, 1, bias=False))
            self.conv.append(nn.BatchNorm2d(filter_size))
            self.conv.append(nn.ReLU())

        self.reset()

    def reset(self):
        for i in self.conv:
            if isinstance(i, nn.Conv2d):
                kaiming_uniform(i.weight)

    def forward(self, input):
        out = input

        for i in self.conv:
            out = i(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, filter_size):
        super().__init__()

        self.conv1 = nn.Conv2d(filter_size, filter_size, [1, 1], 1, 1)
        self.conv2 = nn.Conv2d(filter_size, filter_size, [3, 3], 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(filter_size, affine=False)

        self.reset()

    def forward(self, input, gamma, beta):
        out = self.conv1(input)
        resid = F.relu(out)
        out = self.conv2(resid)
        out = self.bn(out)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = gamma * out + beta

        out = F.relu(out)
        out = out + resid

        return out

    def reset(self):
        kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.zero_()
        kaiming_uniform(self.conv2.weight)

class Classifier(nn.Module):
    def __init__(self, n_input=128, n_filter=512, n_hidden=1024, n_layer=2, n_class=29):
        super().__init__()

        self.conv = nn.Conv2d(n_input, n_filter, [1, 1], 1, 1)
        self.mlp = nn.Sequential(nn.Linear(n_filter, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_class))

        self.reset()

    def reset(self):
        kaiming_uniform(self.conv.weight)
        self.conv.bias.data.zero_()

        for i in self.mlp:
            if isinstance(i, nn.Linear):
                kaiming_uniform(i.weight)
                i.bias.data.zero_()

    def forward(self, input):
        out = self.conv(input)
        b_size, n_filter, _, _ = out.size()
        out = out.view(b_size, n_filter, -1)
        out, _ = out.max(2)
        out = self.mlp(out)

        return out

class FiLM(nn.Module):
    def __init__(self, n_vocab, n_cnn=4, n_resblock=4, conv_hidden=128, embed_hidden=200,
                 gru_hidden=4096, mlp_hidden=256, classes=29):
        super().__init__()

        self.conv = CNN(n_cnn, conv_hidden)
        self.resblocks = nn.ModuleList()
        for i in range(n_resblock):
            self.resblocks.append(ResBlock(conv_hidden))

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.gru = nn.GRU(embed_hidden, gru_hidden, batch_first=True)
        self.film = nn.Linear(gru_hidden, conv_hidden * 2 * n_resblock)

        self.classifier = Classifier(conv_hidden)

        self.n_resblock = n_resblock
        self.conv_hidden = conv_hidden

    def reset(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        kaiming_uniform(self.film.weight)
        self.film.bias.data.zero_()

    def forward(self, image, question, question_len):
        out = self.conv(image)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        _, h = self.gru(embed)
        film = self.film(h.squeeze()).chunk(self.n_resblock * 2, 1)

        for i, resblock in enumerate(self.resblocks):
            out = resblock(out, film[i * 2], film[i * 2 + 1])

        out = self.classifier(out)

        return out