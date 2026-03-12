def model_size_gb(params_billions, bits):
    bytes_per_param = bits / 8
    total_bytes = params_billions * 1e9 * bytes_per_param
    return round(total_bytes / 1e9, 1)

print(model_size_gb(7, 32))   # 28.0
print(model_size_gb(7, 4))    # 3.5
print(model_size_gb(70, 4))   # 35.0
