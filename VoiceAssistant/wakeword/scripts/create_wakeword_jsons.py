"""Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
"""
import os
import json
import random

def main(zero_label_dir, one_label_dir, save_json_path, percent=10):
    zeros = os.listdir(zero_label_dir)
    ones = os.listdir(one_label_dir)
    data = []
    for z in zeros:
        data.append({
            "key": os.path.join(zero_label_dir, z),
            "label": 0
        })
    for o in ones:
        data.append({
            "key": os.path.join(one_label_dir, o),
            "label": 1
        })
    random.shuffle(data)

    train_json_path = os.path.join(save_json_path, "train.json")
    test_json_path = os.path.join(save_json_path, "test.json")

    with open(train_json_path, 'w') as f:
        d = len(data)
        i = 0
        while i < int(d - d / percent):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1

    with open(test_json_path, 'w') as f:
        d = len(data)
        i = int(d - d / percent)
        while i < d:
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1

if __name__ == "__main__":
    # Input values directly
    zero_label_dir = "VoiceAssistant/wakeword/scripts/data/0"  # Example: path to directory with zero labels
    one_label_dir = "VoiceAssistant/wakeword/scripts/data/1"  # Example: path to directory with one labels
    save_json_path = "VoiceAssistant/jsons"  # Example: path to save the JSON files
    percent = 10  # Example: 10 percent for test.json, default is 10
    
    main(zero_label_dir, one_label_dir, save_json_path, percent)
