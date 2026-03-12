def compare_quants(scores_dict):
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1])
    best = sorted_items[0][1]
    print(f"{'Rank':<5} {'Quant':<12} {'Perplexity':<12} {'Degradation'}")
    print("-" * 42)
    for i, (name, score) in enumerate(sorted_items, 1):
        deg = ((score - best) / best) * 100
        print(f"{i:<5} {name:<12} {score:<12.2f} +{deg:.1f}%")

scores = {'Q4_K_M': 5.3, 'F16': 5.1, 'Q8_0': 5.1, 'Q2_K': 7.0}
compare_quants(scores)