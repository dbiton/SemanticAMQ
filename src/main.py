import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

# Your specific imports
from filters.pyfusefilter_filters import Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter
from filters.cuckoo_filter import CuckooFilter
from filters.bloom_filter import BloomFilter
from filters.naive_filter import NaiveFilter
from lsh import CrossPolytopeLSH, SimHashLSH
from util.fetch_datasets import load_embeds

def save_fpr_plot(df, filename="fpr_vs_memory.png"):
    """Saves the False Positive Rate vs. Bits Used plot."""
    plt.figure(figsize=(12, 7))
    # Using hue for the Filter+LSH pair creates a unique line for each
    sns.lineplot(data=df, x='Bits', y='FPR', hue='Config', marker='o')
    plt.title('False Positive Rate (FPR) vs. Bits Used')
    plt.ylabel('FPR (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"FPR plot saved to {filename}")

def save_fnr_plot(df, filename="fnr_vs_memory.png"):
    """Saves the False Negative Rate vs. Bits Used plot."""
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='Bits', y='FNR', hue='Config', marker='s')
    plt.title('False Negative Rate (FNR) vs. Bits Used')
    plt.ylabel('FNR')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"FNR plot saved to {filename}")

def main():
    filters = [Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter, CuckooFilter, BloomFilter]
    lsh_families = [SimHashLSH, CrossPolytopeLSH]
    num_vectors = 1000
    num_vectors_construct = 900
    distance_threshold = 1.0
    dim = 384
    
    # We vary bits to create the 'line' for the plot
    bit_range = [8, 16, 32, 64, 128, 256, 512, 1024] 
    
    dataset_name = 'MsMarco'
    dataset = load_embeds(dataset_name, num_vectors)
    vectors = dataset[0]
    vectors_construct = vectors[:num_vectors_construct]
    vectors_query = vectors[num_vectors_construct:]
    
    # Baseline comparison
    naive_filter = NaiveFilter(dim=dim, distance_threshold=distance_threshold)
    naive_filter.construct(vectors_construct)
    actual_results = naive_filter.query(vectors_query)
    
    results_list = []

    for filter_cls in filters:
        for lsh_cls in lsh_families:
            config_label = f"{filter_cls.__name__} + {lsh_cls.__name__}"
            
            for bits in bit_range:
                print(f"Evaluating {config_label} at {bits} bits...")

                # Initialize LSH with varying size
                if lsh_cls == SimHashLSH:
                    lsh_instance = lsh_cls(dim=dim, num_bits=bits)
                else:
                    # Adjusting rotations to scale memory for CrossPolytope
                    lsh_instance = lsh_cls(dim=dim, num_rotations=max(1, bits // 16))

                filter_instance = filter_cls(dim=dim, lsh=lsh_instance)
                filter_instance.construct(vectors_construct)

                # Query and Metrics
                predicted_results = filter_instance.query(vectors_query)
                
                tp = sum(1 for p, a in zip(predicted_results, actual_results) if p and a)
                fp = sum(1 for p, a in zip(predicted_results, actual_results) if p and not a)
                fn = sum(1 for p, a in zip(predicted_results, actual_results) if not p and a)
                tn = sum(1 for p, a in zip(predicted_results, actual_results) if not p and not a)

                # Metric Definitions
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                results_list.append({
                    'Config': config_label,
                    'Bits': bits,
                    'FPR': fpr,
                    'FNR': fnr
                })

    # Data transformation and plotting
    results_df = pd.DataFrame(results_list)
    save_fpr_plot(results_df)
    save_fnr_plot(results_df)

if __name__ == "__main__":
    main()