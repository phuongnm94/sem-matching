from tree import Tree, Root, Token
import json
from tqdm import tqdm
import unidecode
import argparse

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"

def get_node_info(tree):
    node_info = list_nonterminals(tree.root, [], 1)
    return node_info

def list_nonterminals(cur_node, node_info, node_level):
    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            
            tmp = get_span(child)
            tmp = str(tmp[0]) + ";" + str(tmp[1]-1)
            node_info.append((tmp, node_level,child.label))

    for child in cur_node.children:
        if type(child) != Root and type(child) != Token:
            list_nonterminals(child, node_info, node_level+1)
            
    return node_info
    
def get_span(node):
    return node.get_token_span()

def convert_top_format_to_context_format(gold_filename, without_unspported_label=False):
    result = []
    with open(gold_filename) as gold_file:
        for gold_line in tqdm(gold_file):
            data = gold_line.split("\t")
            context = unidecode.unidecode(data[1])
            label_line =  unidecode.unidecode(data[2])
            if without_unspported_label and "IN:UNSUPPORTED" in label_line:
                continue
            gold_tree = Tree(label_line)
            label = get_node_info(gold_tree)
            result.append({
                "context": context,
                "label": label,
                "org_label": label_line.strip('\n')
            })
    return result

def conver_augumented_data_to_context_format(gold_filename):
    result = []
    with open(gold_filename) as gold_file:
        for gold_line in tqdm(gold_file):
            label_line =  gold_line.strip()
            tmp = [item for item in label_line.split() if not (item[0] == BRACKET_OPEN or item[0] == BRACKET_CLOSE)]
            context = " ".join(tmp)
            gold_tree = Tree(label_line)
            label = get_node_info(gold_tree)
            result.append({
                "context": context,
                "label": label,
                "org_label": label_line.strip('\n')
            })
    return result


def count_label(data):
    label_count = {}
    for item in data:
        for label in item["label"]:
            if label[-1] not in label_count:
                label_count[label[-1]] = 1
            else:
                label_count[label[-1]] += 1
    return label_count

def create_label2id(label_count):
    queries = {"NONE_LABEL": 0}
    for idx, item in enumerate(sorted(label_count.keys())):
        queries[item] = idx + 1
    return queries

def test(data, result):
    for item in data:
        tmp = {}
        for label in item["label"]:
            if label[0] not in tmp:
                tmp[label[0]] = 1
            else:
                tmp[label[0]] += 1
        result.append(max(tmp.values()))
    return result
    
def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--use_augumented_data', action='store_true')
    parser.add_argument('--create_label2id', action='store_true')
    args = parser.parse_args()

    phases = ["train", "dev", "test"]
    tmp = []
    for phase in phases:
        gold_filename = f"data/top/raw/{phase}.tsv"
        result = convert_top_format_to_context_format(gold_filename, without_unspported_label=True)
        if phase == "train" and args.use_augumented_data:
            tmp = conver_augumented_data_to_context_format("data/top/augmented_data_small.txt")
            result += tmp
            write_json(result, f"data/top/extracted_augumented_{phase}.json")
        else:
            write_json(result, f"data/top/extracted_{phase}.json")
        # tmp = test(result, tmp)
        if phase == "train" and args.create_label2id:
            label_count = count_label(result)
            label2id = create_label2id(label_count)
            write_json(label2id, f"data/top/label2id.json")

    # print(max(tmp))

main()