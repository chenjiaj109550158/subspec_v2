import json
import numpy as np
import argparse
from collections import defaultdict

def main(input_path: str, output_path: str, rank_file: str = None):
    """
    Reads a JSON file containing lists of scores for each head,
    calculates the average score for each head, and saves the results
    grouped by layer.

    Optionally, it also generates a ranking of head indices based on
    their average score and saves it to 'rank_file' if provided.
    """
    
    # Step 1: Read and parse the JSON input file
    print(f"Reading data from {input_path}...")
    with open(input_path) as file:
        # Load the entire JSON object
        head_list_data = json.load(file)
        # head_list_data format: {"layer-head": [score1, score2, ...], ...}

    # Step 2: Calculate average scores and organize by layer
    # (temporarily including head_id for sorting)
    print("Calculating average scores and organizing by layer...")
    # layer_wise_scores_temp format: {layer_id: [(head_id, avg_score), ...], ...}
    layer_wise_scores_temp = defaultdict(list)
    
    # Iterate over each item in the dictionary
    # head_key is a string (e.g., "10-5")
    # score_list is a list of scores (e.g., [0.8, 0.9, 0.75])
    for head_key, score_list in head_list_data.items():
        try:
            # Split the "layer-head" key
            layer_str, head_str = head_key.split('-')
            layer_id = int(layer_str)
            head_id = int(head_str)
            
            average_score = 0.0
            # Check if the score list is not empty to avoid errors
            if score_list:
                # Calculate the average score
                average_score = np.mean(score_list)
            
            # Store the (head_id, avg_score) tuple for later sorting
            # Convert to a standard Python float for JSON serialization
            layer_wise_scores_temp[layer_id].append((head_id, float(average_score)))

        except ValueError:
            print(f"Warning: Skipping malformed key: {head_key}")
            continue

    # Step 2.5: Sort heads and convert to the final format
    print("Sorting heads and finalizing format...")
    # final_layer_scores format: {layer_id: [score_head_0, score_head_1, ...]}
    final_layer_scores = {}
    
    # Iterate through each layer
    for layer_id, head_scores_list in layer_wise_scores_temp.items():
        # Sort the list based on head_id (the first element of the tuple)
        sorted_heads = sorted(head_scores_list, key=lambda x: x[0])
        
        # Extract only the scores to create the new list
        final_scores_list = [score for head_id, score in sorted_heads]
        
        # Store the final list of scores
        final_layer_scores[layer_id] = final_scores_list

    # Step 3: Save the results to a JSON file
    print(f"Saving layer-grouped scores to {output_path}...")
    with open(output_path, 'w') as outfile:
        # 'indent=4' makes the output JSON file human-readable
        # 'sort_keys=True' ensures layer_ids are stored in numerical order
        json.dump(final_layer_scores, outfile, indent=4, sort_keys=True)
    
    print(f"Successfully saved average SRH scores (grouped by layer) to: {output_path}")

    # --- 
    # Step 4: [NEW] Generate and save the head ranking file (if requested)
    # ---
    if rank_file:
        print(f"\nGenerating head ranking data (by score)...")
        # ranked_head_indices format: {layer_id: [ranked_head_id_1, ranked_head_id_2, ...]}
        ranked_head_indices = {}

        # Iterate over the {layer_id: [score_head_0, ...]} dictionary
        for layer_id, scores_list in final_layer_scores.items():
            
            # 1. Create a list of tuples: [(head_id, score), ...]
            #    (enumerate gives the index, which is the head_id)
            head_score_pairs = list(enumerate(scores_list))
            
            # 2. Sort the list by score (x[1]) in descending order (reverse=True)
            #    This meets the "分數越高排名的數字越低" requirement
            sorted_pairs = sorted(head_score_pairs, key=lambda x: x[1], reverse=True)
            
            # 3. Extract just the head_ids (x[0]) from the sorted list
            ranked_indices = [head_id for head_id, score in sorted_pairs]
            
            # 4. Store the ranked list of head indices
            ranked_head_indices[layer_id] = ranked_indices

        # 5. Save the ranking data to the new JSON file
        print(f"Saving head ranking (by score) to {rank_file}...")
        with open(rank_file, 'w') as outfile:
            json.dump(ranked_head_indices, outfile, indent=4, sort_keys=True)
        
        print(f"Successfully saved head ranking data to: {rank_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='get_individual_head_scores.py',
        description='Extracts the average SRH score for each head from a JSON file and saves them grouped by layer into a new JSON file. Can optionally also save a head ranking file.'
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        default='specdecodes/models/utils/compresskv/scores/Qwen2.5-0.5B-Instruct_SRH.json',
        help='Path to the input JSON file containing the raw head score lists.'
    )
    parser.add_argument(
        'output_path',
        nargs='?',
        # Change the default output filename to reflect its content
        default='specdecodes/models/utils/compresskv/scores/Qwen2.5-0.5B-Instruct_list_SRH.json',
        help='Path to the output JSON file for storing the average head scores (grouped by layer).'
    )
    # ---
    # [NEW] Added a new optional argument for the rank file
    # ---
    parser.add_argument(
        '--rank_file',
        type=str,
        default='specdecodes/models/utils/compresskv/scores/Qwen2.5-0.5B-Instruct_head_idx.json', # By default, this feature is off
        help='(Optional) Path to the output JSON file for storing the *ranked head indices* (by score).'
    )
    
    args = parser.parse_args()
    
    # Pass the new argument to the main function
    main(args.input_path, args.output_path, args.rank_file)