import os.path
import pickle
from collections import Counter
from typing import Dict, List

import nltk
from pycocotools.coco import COCO
from tqdm import tqdm


class Vocabulary:
    def __init__(self,
                 word2idx: Dict[str, int],
                 start_token: str,
                 stop_token: str,
                 unknown_token: str,
                 the_longest_caption: int) -> None:
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.start_token = start_token
        self.stop_token = stop_token
        self.unknown_token = unknown_token
        self.the_longest_caption = the_longest_caption

    def tokenize_caption(self, caption: str) -> List[int]:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        output = [self.word2idx[self.start_token],
                  *[self.word2idx.get(token, self.word2idx[self.unknown_token]) for token in tokens],
                  self.word2idx[self.stop_token]]
        return output + [self.word2idx[self.stop_token]] * (self.the_longest_caption+2-len(output))

    def token_to_caption(self, tokens: List[int]) -> List[str]:
        return [self.idx2word[token] for token in tokens if token not in [self.word2idx[self.start_token],
                                                                          self.word2idx[self.stop_token],
                                                                          self.word2idx[self.unknown_token]
                                                                          ]]

    def __len__(self) -> int:
        return len(self.idx2word)


class VocabularyFactory:

    @staticmethod
    def build_vocabulary_from_annotation(
            vocab_threshold: int,
            vocab_output_filepath: str,
            annotation_filepath: str,
            start_token: str = '<START>',
            stop_token: str = '<STOP>',
            unknown_token: str = '<UNKNOWN>') -> None:
        if not os.path.exists(annotation_filepath):
            raise IOError(f'Path {annotation_filepath} is not valid')

        coco = COCO(annotation_filepath)
        counter = Counter()
        the_longest_caption = 0

        # Count words
        for annotations in tqdm(coco.anns.values()):
            caption = annotations['caption']
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            if len(tokens) > the_longest_caption:
                the_longest_caption = len(tokens)

        word2idx = {
            start_token: 0,
            stop_token: 1,
            unknown_token: 2,
            **{word: idx for idx, (word, occurrences) in enumerate(counter.most_common(), 3) if occurrences > vocab_threshold}
        }

        vocab = Vocabulary(
            word2idx=word2idx,
            start_token=start_token,
            stop_token=stop_token,
            unknown_token=unknown_token,
            the_longest_caption=the_longest_caption
        )

        with open(vocab_output_filepath, 'wb') as f:
            pickle.dump(vocab, f)

    @staticmethod
    def build_vocabulary_from_file(
            vocab_filepath
    ) -> Vocabulary:
        with open(vocab_filepath, 'rb') as f:
            return pickle.load(f)
