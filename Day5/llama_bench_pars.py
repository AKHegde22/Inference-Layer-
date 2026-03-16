def parse_bench_output(text):
    result = {}
    for line in text.strip().splitlines():
        if "|" not in line or "---" in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 7:
            continue
        test = cells[5]
        tps_raw = cells[6].split(" ± ")[0].strip()
        try:
            tps = float(tps_raw)
        except ValueError:
            continue
        if test.startswith("pp"):
            result["pp_tokens_per_sec"] = tps
            result["backend"] = cells[3]
            result["ngl"] = int(cells[4])
            result["model_size_gib"] = float(cells[1].split()[0])
        elif test.startswith("tg"):
            result["tg_tokens_per_sec"] = tps
    return result

# Replace with the fucntion so that you get the actual value
sample = """
| model | size | params | backend | ngl | test | t/s |
|---|---|---|---|---|---|---|
| mistral 7B Q4_K_M | 4.06 GiB | 7.24 B | CUDA | 99 | pp512 | 2340.50 ± 12.3 |
| mistral 7B Q4_K_M | 4.06 GiB | 7.24 B | CUDA | 99 | tg128 | 87.23 ± 0.8 |
"""
print(parse_bench_output(sample))