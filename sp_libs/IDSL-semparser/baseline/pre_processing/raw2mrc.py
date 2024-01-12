#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: genia2mrc.py

import os
import json
import argparse

# Raw data format:
# {
#     "context": "我是一个中国人",
#     "label": {
#         "PER": ["0;3"],
#         "LOC": ["4;7"]
#     }
# }

def convert_file(input_file, output_file):
    """
    Convert raw data to MRC format
    """
    all_data = json.load(open(input_file))
    output = []
    for idx, data in enumerate(all_data):
        context = data["context"]
        label2positions = data["label"]
        start_position, end_position, entity_level, label_type = [], [], [], []
        for label in label2positions:
            start, end = label[0].split(";")
            start_position.append(int(start))
            end_position.append(int(end))
            entity_level.append(label[1])
            label_type.append(label[-1])

        # start_position, end_position, entity_level, label_type = zip(*sorted(zip(start_position, end_position, entity_level, label_type), key=lambda x: (x[0], x[1])))

        mrc_sample = {
            "context": context,
            "entity_level": entity_level,
            "label_type": label_type,
            "start_position": start_position,
            "end_position": end_position,
            "qas_id": idx,
            "org_label": data["org_label"]
        }
        output.append(mrc_sample)
    json.dump(output, open(output_file, "w"), ensure_ascii=False, indent=2)
    print(f"Create {len(output)} samples and save to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--use_augumented_data', action='store_true')
    args = parser.parse_args()

    base_dir = "data/top"
    os.makedirs(base_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        if phase == "train" and args.use_augumented_data:
            old_file = os.path.join(base_dir, f"extracted_augumented_{phase}.json")
            new_file = os.path.join(base_dir, f"mrc-ner-augumented.{phase}")
        else:
            old_file = os.path.join(base_dir, f"extracted_{phase}.json")
            new_file = os.path.join(base_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file)

if __name__ == '__main__':
    main()
