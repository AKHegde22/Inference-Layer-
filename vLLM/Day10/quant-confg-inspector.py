import json

class QuantInspector:
    VLLM_SUPPORTED = {None, "awq", "gptq", "fp8", "bitsandbytes", "squeezellm"}

    def __init__(self):
        self.config = {}

    def load_config(self, model_path_or_dict):
        if isinstance(model_path_or_dict, dict):
            self.config = model_path_or_dict
        else:
            with open(model_path_or_dict) as f:
                self.config = json.load(f)

    def get_quant_info(self):
        qc = self.config.get("quantization_config", {})
        fmt = qc.get("quant_type") or qc.get("quant_method") or None
        if fmt:
            fmt = fmt.lower()
        return {
            "is_quantized":   bool(qc),
            "format":         fmt,
            "bits":           qc.get("bits"),
            "group_size":     qc.get("group_size"),
            "zero_point":     qc.get("zero_point"),
            "dtype":          self.config.get("torch_dtype", "unknown"),
            "architecture":   (self.config.get("architectures") or ["unknown"])[0],
            "vocab_size":     self.config.get("vocab_size"),
            "context_length": self.config.get("max_position_embeddings"),
        }

    def summarize(self):
        info = self.get_quant_info()
        print("=" * 46)
        print(f"  Model Config Summary")
        print("=" * 46)
        print(f"  {'Architecture':<20} {info['architecture']}")
        print(f"  {'dtype':<20} {info['dtype']}")
        print(f"  {'Vocab Size':<20} {info['vocab_size']}")
        print(f"  {'Context Length':<20} {info['context_length']}")
        print(f"  {'Quantized':<20} {info['is_quantized']}")
        if info["is_quantized"]:
            print(f"  {'Format':<20} {info['format']}")
            print(f"  {'Bits':<20} {info['bits']}")
            print(f"  {'Group Size':<20} {info['group_size']}")
            print(f"  {'Zero Point':<20} {info['zero_point']}")
        print("=" * 46)

    def is_compatible_with_vllm(self):
        info = self.get_quant_info()
        fmt  = info["format"]
        if fmt not in self.VLLM_SUPPORTED:
            print(f"WARNING: format '{fmt}' is not supported by vLLM.")
            return False
        return True


awq_config = {
    "architectures": ["MistralForCausalLM"],
    "torch_dtype": "float16",
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "quantization_config": {
        "quant_type": "awq", "bits": 4,
        "group_size": 128, "zero_point": True, "version": "GEMM"
    }
}
fp16_config = {
    "architectures": ["LlamaForCausalLM"],
    "torch_dtype": "bfloat16",
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
}

insp = QuantInspector()
for cfg in [awq_config, fp16_config]:
    insp.load_config(cfg)
    insp.summarize()
    print("vLLM compatible:", insp.is_compatible_with_vllm())
    print()
