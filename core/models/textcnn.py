import torch.nn.functional as F
from torch import nn
import torch
from torch.autograd import Variable
from core.vectorizers import IndexVectorizer


class MultiKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelConv, self).__init__()
        out_channels //= 3
        self.conv3x1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv4x1 = nn.Conv1d(in_channels, out_channels, kernel_size=4, padding=2)
        self.conv5x1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        f1 = F.relu(self.conv3x1(x))
        f2 = F.relu(self.conv4x1(x)[:, :, :-1])
        f3 = F.relu(self.conv5x1(x))
        return torch.cat([f1, f2, f3], dim=1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, maxlen,
                 embeddings=None, freeze_embeddings=False):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.conv = MultiKernelConv(embedding_dim, 24)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(24 * maxlen//2, 32)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embedding(x)  # N, word, channel (32, 104, 50)
        x = x.permute(0, 2, 1)  # swap to N, channel, word (32, 50, 104)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class TextCNNWrapper:
    def __init__(self):
        self.class_map = ('subjective', 'objective')
        checkpoint = torch.load('core/models/state_dicts/TextCNN_5epoch_glove.pth.tar')
        self.vectorizer = IndexVectorizer(min_frequency=4, maxlen=104)
        self.vectorizer.build_from_vocabulary(checkpoint['vectorizer_state'])
        self.model = TextCNN(4233, 50, 104)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def prepare_text(self, text):
        text = text.lower().split(" ")
        vectorized_text = self.vectorizer.transform_document(text)
        return Variable(torch.Tensor(vectorized_text)).unsqueeze(0).long()

    def classify(self, text):
        vectorized_text = self.prepare_text(text)
        out = F.softmax(self.model(vectorized_text)).data
        print(out)
        prob, pred = torch.max(out, dim=1)
        return round(prob[0], 4), self.class_map[pred[0]]

