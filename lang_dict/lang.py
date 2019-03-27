from config import cfg


class LanguageIndex():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = cfg.CHAR_VECTOR

        self.create_index()

    def create_index(self):
        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2
        self.word2idx[''] = 3
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 4

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

