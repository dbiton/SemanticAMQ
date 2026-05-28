import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

# Your specific imports
from filters.dummy_filter import DummyFilter
from filters.pyfusefilter_filters import Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter
from filters.cuckoo_filter import CuckooFilter
from filters.lsh_bloom import LSHBloom
from filters.bloom_filter import BloomFilter
from filters.naive_filter import NaiveFilter
from filters.quotient_filter import QuotientFilterWrapper
from lsh import CrossPolytopeLSH, SimHashLSH
from util.fetch_datasets import load_embeds

def save_cost_plot(df, filename="cost_vs_memory.png"):
    """Saves the Cost vs. Bits Used plot."""
    plt.figure()
    # Using hue for the Filter+LSH pair creates a unique line for each
    df['Cost'] = df['FPR'] * 10 + df['FNR']
    sns.lineplot(data=df, x='Bits', y='Cost', hue='Config', marker='o')
    plt.ylabel('Cost')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Cost plot saved to {filename}")

def save_fpr_plot(df, filename="fpr_vs_memory.png"):
    """Saves the False Positive Rate vs. Bits Used plot."""
    plt.figure()
    # Using hue for the Filter+LSH pair creates a unique line for each
    sns.lineplot(data=df, x='Bits', y='FPR', hue='Config', marker='o')
    plt.ylabel('FPR')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"FPR plot saved to {filename}")

def save_fnr_plot(df, filename="fnr_vs_memory.png"):
    """Saves the False Negative Rate vs. Bits Used plot."""
    plt.figure()
    sns.lineplot(data=df, x='Bits', y='FNR', hue='Config', marker='s')
    plt.ylabel('FNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"FNR plot saved to {filename}")

def main():
    filters = [Fuse16Filter, LSHBloom, DummyFilter]
    lsh_families = [CrossPolytopeLSH]
    num_vectors = 1000
    num_vectors_construct = 900
    distance_threshold = 1.0
    dim = 384
    
    # We vary bits to create the 'line' for the plot
    bit_range = [16, 64, 512] 
    
    dataset_name = 'StackOverflow'
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

                # Initialize LSH with varying size based on the current budget
                lsh_instance = lsh_cls.build_optimized(dim=dim, threshold_d=distance_threshold, K=None, max_L=bits)

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
                
                print(f"Results for {config_label} at {bits} bits: FPR={fpr:.4f}, FNR={fnr:.4f}, Bytes={filter_instance.get_bytesize()}")   

                result = {
                    'Config': config_label,
                    'Bits': bits,
                    'FPR': fpr,
                    'FNR': fnr,
                    'Bytes': filter_instance.get_bytesize()
                }

                results_list.append(result)

    # Data transformation and plotting
    results_df = pd.DataFrame(results_list)
    save_fpr_plot(results_df)
    save_fnr_plot(results_df)
    save_cost_plot(results_df)

if __name__ == "__main__":
    main()