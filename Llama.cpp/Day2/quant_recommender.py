def recommend_quant(vram_gb, use_case='chat'):
    if vram_gb < 4:
        return "Q2_K"
    elif vram_gb < 5:
        return "Q3_K_M"
    elif vram_gb < 6:
        return "Q4_K_M"
    elif vram_gb < 8:
        return "Q6_K" if use_case == "coding" else "Q5_K_M"
    else:
        return "Q8_0"

print(recommend_quant(4, 'chat'))    # Q3_K_M
print(recommend_quant(6, 'coding'))  # Q6_K
print(recommend_quant(16, 'chat'))   # Q8_0