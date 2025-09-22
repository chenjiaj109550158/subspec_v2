# run_benchmark.py
import os
import shutil
import json
import time
import torch
import random
import logging
import gc
from tqdm import tqdm
import numpy as np

from .benchmarks.utils.eval_acc import run_gsm8k_eval, run_aime_eval, run_livecodebench_eval, run_mmlu_pro_eval
from .benchmarks.gsm8k import load_gsm8k_dataset_answer
from .benchmarks.aime import load_aime_dataset_answer
from .benchmarks.livecodebench import load_livecodebench_dataset_answer
from .benchmarks.mmlu_pro import load_mmlu_pro_dataset_answer

DATASET_LOADER = {
    "gsm8k":      load_gsm8k_dataset_answer,
    "aime":       load_aime_dataset_answer,
    "livecodebench": load_livecodebench_dataset_answer,
    "mmlu_pro":   load_mmlu_pro_dataset_answer,
}

BENCHMARK_EVALUATORS = {
    "gsm8k":      run_gsm8k_eval,
    "aime":       run_aime_eval,
    "livecodebench": run_livecodebench_eval,
    "mmlu_pro":   run_mmlu_pro_eval,
}

def main(builder, benchmarks=None, max_samples=None):
    torch.manual_seed(0)
    random.seed(0)
        
    # Enable profiling, disable logging profiling results
    builder.generator_profiling = True
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # Build bench_list and check if all names are valid
    bench_list = benchmarks.split(",") if benchmarks is not None else []
    for b in bench_list:
        if b not in DATASET_LOADER:
            raise ValueError(f"Unknown benchmark: {b}. Available benchmarks: {list(DATASET_LOADER.keys())}")
    print(f"Benchmarks to run: {bench_list}")
    
    # Handle output directories
    if args.out_dir is not None:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        print(f"Deleted old {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        
    # Run benchmarks
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        # fix random seed to 0 for each benchmark for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Handle output directories
        log_dir = os.path.join(log_dir_base, bench_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")
        
        # Load dataset
        if not bench_name == "mmlu_pro":
            dataset = DATASET_LOADER[bench_name]()
            num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
            print(f"Running benchmark: {bench_name}, samples: {num_samples}")
    
            random.shuffle(dataset)
            dataset = dataset[:num_samples]
        else:
            # MMLU-Pro dataset is loaded differently
            dataset = load_mmlu_pro_dataset_answer(sample_per_category=max_samples, seed=0)
            num_samples = len(dataset)
            print(f"Running benchmark: {bench_name}, samples: {num_samples}")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    
        # Evaluate
        tput_mean, tput_std, acc_rate_mean, acc_rate_std, accuracy, avg_draft_time, avg_target_time, peak_mem = \
            BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        if hasattr(generator, "judge_acc_len_list"):
            # print("acc_list:", generator.judge_acc_len_list)
            tacc_judge_value = np.mean(generator.judge_acc_len_list)
        else:
            tacc_judge_value = 0

        # Write results to file
        with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
            json.dump({
                bench_name: {
                    "tput":         f"{tput_mean:.3f}",
                    "tput_std":     f"{tput_std:.3f}",
                    "Tacc":         f"{acc_rate_mean:.3f}",
                    "Tacc_std":     f"{acc_rate_std:.3f}",
                    "Accuracy":     f"{accuracy:.3f}",         
                    "avg_draft_time":  f"{avg_draft_time:.3f}",
                    "avg_target_time": f"{avg_target_time:.3f}",
                    "peak_memory":     f"{peak_mem:.3f} GiB",
                    "Tacc_judge" : f"{tacc_judge_value:.3f}",
                }
            }, f, indent=4)
            f.write("\n")
