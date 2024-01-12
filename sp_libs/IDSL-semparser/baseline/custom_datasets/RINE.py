#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

class RINEDataset(Dataset):
    """
    RINE Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
    """
    def __init__(self, json_path, label2id_path, tokenizer: AutoTokenizer, max_length: int = 512):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = json.load(open(label2id_path, encoding="utf-8"))
        self.label2id["[EOP]"] = len(self.label2id.values())
        self.processed_data = self.process_data(self.all_data)

    @staticmethod
    def update_pos(start_positions, end_positions, cursor):
        new_start_positions = [start+1 if start >= cursor else start for start in start_positions ]
        new_end_positions = [end+1 if end >= cursor else end for end in end_positions ]
        return new_start_positions, new_end_positions
        
    def process_data(self, data):
        data_dict = {}
        print("Processing data...")
        for item in tqdm(data):
            context = item["context"]
            start_positions = item["start_position"]
            end_positions = item["end_position"]
            labels = item["label_type"]
            entity_levels = item["entity_level"]
            cur_words = context.split()
            num_entities = len(start_positions)
            for idx in range(num_entities):
                if entity_levels[idx] not in data_dict:
                    data_dict[entity_levels[idx]] = []
                # print("-"*50)
                # print("cur_words:", " ".join(cur_words))
                # print("start_positions:", start_positions[idx])
                # print("end_positions:", end_positions[idx])
                # print("labels:", labels[idx])
                data_dict[entity_levels[idx]].append(self.create_training_tensors(" ".join(cur_words), start_positions[idx], \
                                                    end_positions[idx], labels[idx]))

                # Update context and postions
                cur_words.insert(end_positions[idx]+1, f"]")
                cur_words.insert(start_positions[idx], f"[{labels[idx]}")
                start_positions, end_positions = self.update_pos(start_positions, end_positions, start_positions[idx])
                start_positions, end_positions = self.update_pos(start_positions, end_positions, end_positions[idx]+1)

            if entity_levels[-1] + 1 not in data_dict:
                data_dict[entity_levels[-1] + 1] = []
            data_dict[entity_levels[-1] + 1].append(self.create_training_tensors(" ".join(cur_words), -1, -1, "[EOP]"))
            # print(data_dict[entity_levels[-1] + 1])

        result = []
        for value in data_dict.values():
            result += value
        return result

    def create_training_tensors(self, context, start, end, label):
         # query = "Find the intents and slots of the sentence."
        start_positions = [start] if start != -1 else []
        end_positions = [end] if end != -1 else []

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        sample_tokens = self.tokenizer(context, return_token_type_ids=True, return_offsets_mapping=True)
        tokens = sample_tokens.input_ids
        type_ids = sample_tokens.token_type_ids
        offsets = sample_tokens.offset_mapping

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            if token_start not in origin_offset2token_idx_start:
                origin_offset2token_idx_start[token_start] = token_idx
            if token_end not in origin_offset2token_idx_end:
                origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        # print(tokens)
        # print(self.tokenizer.convert_ids_to_tokens(tokens))
        # print(new_start_positions)
        # print(new_end_positions)
        label_mask = [
            (0 if offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]        

        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        for token_idx in range(len(tokens)):
            current_word_idx = sample_tokens.words()[token_idx]
            next_word_idx = sample_tokens.words()[token_idx+1] if token_idx+1 < len(tokens) else None
            prev_word_idx = sample_tokens.words()[token_idx-1] if token_idx-1 > 0 else None
            if prev_word_idx is not None and current_word_idx == prev_word_idx:
                start_label_mask[token_idx] = 0
            if next_word_idx is not None and current_word_idx == next_word_idx:
                end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is </s>
        sep_token = 2 # SEP token id
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        # Label types
        label_type = [self.label2id[label]]
        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            torch.LongTensor(label_type)
        ]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
        """
        return self.processed_data[item]

def run_dataset():
    """test dataset"""
    import os
    from custom_datasets.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader

    # en datasets   
    bert_path = "/home/s2110418/Master/Semantic_parsing/mrc-for-flat-nested-ner/roberta/roberta-large"
    json_path = "/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams/data/top/mrc-ner.train"
    label2id_path = "/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_RINE/data/top/label2id.json"
    is_chinese = False

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    dataset = RINEDataset(json_path=json_path, label2id_path=label2id_path, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=2,
                            collate_fn= lambda b: collate_to_max_length(b, ignore_index=-100, pad_id=1))

    import  numpy as np
    for batch in dataloader:
        # pass
        print(batch)
        break 

if __name__ == '__main__':
    run_dataset()
