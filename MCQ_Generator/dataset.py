from datasets import load_dataset

race = load_dataset("race", "all")
print("RACE downloaded:", race)

boolq = load_dataset("boolq")
print("BoolQ downloaded:", boolq)

openbookqa = load_dataset("openbookqa", "main")
print("OpenBookQA downloaded:", openbookqa)

arc_easy = load_dataset("ai2_arc", "ARC-Easy")
arc_challenge = load_dataset("ai2_arc", "ARC-Challenge")
print("ARC-Easy and ARC-Challenge downloaded.")