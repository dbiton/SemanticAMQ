from filters.pyfusefilter_filters import Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter
from filters.naive_filter import NaiveFilter
from lsh import CrossPolytopeLSH, SimHashLSH
import numpy as np
from util.fetch_datasets import load_embeds

def main():
    filters = [Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter]
    lsh_families = [SimHashLSH, CrossPolytopeLSH]
    num_vectors = 50000
    num_vectors_construct = 45000
    distance_threshold = 1.0
    assert num_vectors_construct <= num_vectors, "num_vectors_construct must be less than or equal to num_vectors"
    dim = 384
    
    dataset_name = 'MsMarco'
    dataset = load_embeds(dataset_name, num_vectors)
    vectors = dataset[0]
    vectors_construct = vectors[:num_vectors_construct]
    vectors_query = vectors[num_vectors_construct:]
    
    naive_filter = NaiveFilter(dim=dim, distance_threshold=distance_threshold)
    naive_filter.construct(vectors_construct)
    actual_results = naive_filter.query(vectors_query)
    
    for filter in filters:
        for lsh in lsh_families:
            print(f"Testing {filter.__name__} with {lsh.__name__}")

            lsh_instance = lsh(dim=dim, num_bits=64) if lsh == SimHashLSH else lsh(dim=dim, num_rotations=64)
            filter_instance = filter(dim=dim, lsh=lsh_instance)

            filter_instance.construct(vectors_construct)

            # Test querying a random vector
            predicted_results = filter_instance.query(vectors_query)
            TP = sum(1 for p, a in zip(predicted_results, actual_results) if p and a)
            FP = sum(1 for p, a in zip(predicted_results, actual_results) if p and not a)
            FN = sum(1 for p, a in zip(predicted_results, actual_results) if not p and a)
            TN = sum(1 for p, a in zip(predicted_results, actual_results) if not p and not a)
            print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

if __name__ == "__main__":
    main()
