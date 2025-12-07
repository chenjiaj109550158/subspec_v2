from datasets import load_dataset

def load_longbench_v2_dataset_answer():
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    data = []
    for item in ds:
        data.append({
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
            "retrieved_context": item.get("retrieved_context", []),
        })
    return data