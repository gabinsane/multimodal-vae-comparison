from typing import List, Set, Tuple
import numpy as np


def create_commands(command_type: str,
                    property_tuple: Tuple,
                    action_verbs: List[str] = [],
                    use_base=True,
                    use_synonyms=False) -> List[str]:
    sentences = []
    primary_property, secondary_property = property_tuple
    if command_type == 'instruction':
        assert len(action_verbs) > 0
        beginnings = action_verbs
    elif command_type == 'repair':
        beginnings = ['no i meant', 'no', 'sorry', 'pardon', 'excuse me', 'actually']
        # no i meant the red
        # no i meant the red cube
    elif command_type == 'negation':
        # not the red
        # not the red cube
        beginnings = ['not']
    else:
        raise ValueError('Unknown command type')

    for _begin in beginnings:
        if use_base:
            sentences.append(_begin + " the " + primary_property.name.lower() + " " + secondary_property.name.lower())
        if use_synonyms:
            # combine synonyms with synonyms
            for psyn in primary_property.value[1]:
                for ssyn in secondary_property.value[1]:
                    sentences.append(_begin + " the " + psyn.lower() + " " + ssyn.lower())
        if use_base and use_synonyms:
            # combine property name with synonyms
            for psyn in primary_property.value[1]:
                sentences.append(_begin + " the " + psyn.lower() + " " + secondary_property.name.lower())
            for ssyn in secondary_property.value[1]:
                sentences.append(_begin + " the " + primary_property.name.lower() + " " + ssyn.lower())
    return list(set(sentences))


def parse_instructions(instructions: List[str]) -> Tuple[Set[str], int]:
    word_list = []
    max_instruction_len = 0
    for _instrucion in instructions:
        _splitted = _instrucion.lower().split(' ')
        if len(_splitted) > max_instruction_len:
            max_instruction_len = len(_splitted)
        word_list.extend(_splitted)
    return set(word_list), max_instruction_len


def word_in_string(instr_string: str, word_lst: np.ndarray):
    match_array = np.array([word in instr_string for word in word_lst]).astype(int)
    # additional check, because argmax of zero vector yields result of 0
    if match_array.sum():
        word_idx = np.argmax(match_array)
        return word_lst[word_idx]
    return ''


class Vocabulary:

    def __init__(self, words: List[str]):
        word_list = ['<pad>'] + sorted(list(set(words)))
        _idx_list = np.arange(0, len(word_list))
        self.idx2word = dict(zip(_idx_list, word_list))
        self.word2idx = dict(zip(word_list, _idx_list))
        assert len(self.idx2word) == len(self.word2idx)

    def idx_to_word(self, idx: int) -> str:
        return self.idx2word[idx]

    def word_to_idx(self, word: str) -> int:
        return self.word2idx[word]

    def __call__(self, word) -> int:
        return self.word_to_idx(word)

    def __len__(self) -> int:
        return len(self.word2idx)
