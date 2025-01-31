# -*- coding: utf-8 -*-


class Corpus(object):
    UNK = '<UNK>'

    def __init__(self, fdata):
        # Sentences for obtaining data
        self.sentences = self.preprocess(fdata)
        # Words, tags, chars
        self.words, self.tags, self.chars = self.parse(self.sentences)
        # Add unknown character
        self.chars += [self.UNK]

        # Dictionary of words, tags, chars
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # Dictionary of tags
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        # Dictionary of chars
        self.cdict = {c: i for i, c in enumerate(self.chars)}

        # Unknown index
        self.ui = self.cdict[self.UNK]

        # Sentence number
        self.ns = len(self.sentences)
        # Word number
        self.nw = len(self.words)
        # Tag number
        self.nt = len(self.tags)
        # Char number
        self.nc = len(self.chars)

    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wiseq = [
                tuple(self.cdict.get(c, self.ui) for c in w)
                for w in wordseq
            ]
            tiseq = [self.tdict[t] for t in tagseq]
            data.append((wiseq, tiseq))
        return data

    def __repr__(self):
        info = "%s(\n" % self.__class__.__name__
        info += "  num of sentences: %d\n" % self.ns
        info += "  num of words: %d\n" % self.nw
        info += "  num of tags: %d\n" % self.nt
        info += "  num of chars: %d\n" % self.nc
        info += ")"
        return info

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    @staticmethod
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(w for wordseq in wordseqs for w in wordseq))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        chars = sorted(set(''.join(words)))
        return words, tags, chars
