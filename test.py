import torch
import time
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub
import platform

# 尝试导入 flashinfer，如果失败则禁用相关测试
try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print(
        "⚠️  Warning: flashinfer is not installed. Skipping FlashInfer RMSNorm benchmark."
    )
    print("   Please install it via: pip install flashinfer")


# --- 实现 1: Qwen2 Fused RMSNorm ---
@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 这是当融合核心加载失败时的备用Python实现
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# --- 实现 2: PyTorch Native RMSNorm ---
class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states):
        return self.norm(hidden_states)


# --- 实现 3: PyTorch LayerNorm ---
class TorchLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states):
        return self.norm(hidden_states)


# --- 实现 4: FlashInfer Fused RMSNorm ---
class FlashInferRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 确保 flashinfer 可用，否则这个类无法工作
        if not FLASHINFER_AVAILABLE:
            raise ImportError(
                "FlashInferRMSNorm requires the 'flashinfer' package to be installed."
            )
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # flashinfer.norm.rmsnorm 需要一个2D输入 (num_tokens, hidden_size)
        # 我们的输入是3D的 (batch_size, seq_len, hidden_size)
        # 所以我们需要先 reshape
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]

        # 使用 view 来创建一个2D的视图，这通常是零成本操作
        reshaped_input = hidden_states.view(-1, hidden_size)

        # 调用高度优化的 flashinfer 核心
        normalized_output = flashinfer.norm.rmsnorm(
            reshaped_input, self.weight, self.eps
        )

        # 将输出 reshape 回原来的3D形状
        return normalized_output.view(original_shape)


# --- 基准测试函数 ---
def benchmark(model, x, num_runs=200, warmup_runs=20):
    try:
        # Warmup
        for _ in range(warmup_runs):
            _ = model(x)

        if x.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(x)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) / num_runs * 1000
        return avg_time_ms
    except Exception as e:
        # 如果模型初始化或执行失败，返回一个标记值
        # print(f"Benchmarking failed for {model.__class__.__name__}: {e}")
        return float("inf")


# --- 新增：输出结果对比函数 ---
def compare_outputs(reference_model, models_dict, x, config_name):
    """比较各种模型输出与参考模型的差异"""
    print(f"\n📊 Output Comparison for {config_name}")
    print(
        f"{'Implementation':<20} | {'Max Abs Diff':<15} | {'L2 Distance':<15} | {'Mean Abs Diff':<15}"
    )
    print("-" * 75)

    with torch.no_grad():
        reference_output = reference_model(x)

        for name, model in models_dict.items():
            try:
                output = model(x)

                # 计算各种差异指标
                abs_diff = torch.abs(output - reference_output)
                max_abs_diff = torch.max(abs_diff).item()
                l2_distance = torch.norm(output - reference_output).item()
                mean_abs_diff = torch.mean(abs_diff).item()

                print(
                    f"{name:<20} | {max_abs_diff:<15.2e} | {l2_distance:<15.2e} | {mean_abs_diff:<15.2e}"
                )

            except Exception as e:
                print(f"{name:<20} | {'ERROR':<15} | {'ERROR':<15} | {'ERROR':<15}")


# --- 更新的基准测试函数 ---
def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("-" * 50)
    print(f"Running on Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Platform: {platform.system()}")
    print("-" * 50)

    test_configs = [
        (1, 128, 768),
        (4, 512, 1024),
        (8, 2048, 1024),
        (1, 1024, 1024),
        (1, 8192, 1024),
    ]

    # 存储所有测试结果
    all_results = []

    # === 第一部分：性能测试 ===
    header = f"{'Config (bsz, seq, hidden)':<30} | {'PyTorch RMSNorm':<18} | {'Qwen2 RMSNorm':<18} | {'FlashInfer RMSNorm':<20} | {'PyTorch LayerNorm':<20} | {'Best Speedup':<12}"
    print("\n🚀 Performance Benchmark (ms)")
    print(header)
    print("-" * (len(header) + 5))

    for bsz, seq_len, hidden_size in test_configs:
        config_str = f"({bsz}, {seq_len}, {hidden_size})"
        x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

        # 初始化模型
        qwen_norm = Qwen2RMSNorm(hidden_size).to(device=device, dtype=dtype)
        torch_rms_norm = TorchRMSNorm(hidden_size).to(device=device, dtype=dtype)
        torch_layer_norm = TorchLayerNorm(hidden_size).to(device=device, dtype=dtype)

        # 性能测试
        flashinfer_time = float("inf")
        flashinfer_norm = None
        if FLASHINFER_AVAILABLE:
            flashinfer_norm = FlashInferRMSNorm(hidden_size).to(
                device=device, dtype=dtype
            )
            flashinfer_time = benchmark(flashinfer_norm, x)

        torch_rms_time = benchmark(torch_rms_norm, x)
        qwen_time = benchmark(qwen_norm, x)
        torch_layer_time = benchmark(torch_layer_norm, x)

        times = [
            t
            for t in [torch_rms_time, qwen_time, flashinfer_time, torch_layer_time]
            if t != float("inf")
        ]
        if not times:
            continue

        min_time = min(times)
        max_time = max(times)
        best_speedup = max_time / min_time if min_time > 0 else 0

        flash_str = (
            f"{flashinfer_time:<20.4f}" if FLASHINFER_AVAILABLE else f"{'N/A':<20}"
        )

        print(
            f"{config_str:<30} | {torch_rms_time:<18.4f} | {qwen_time:<18.4f} | {flash_str} | {torch_layer_time:<20.4f} | {best_speedup:.2f}x"
        )

        # 保存结果用于精度测试
        all_results.append(
            {
                "config": config_str,
                "x": x,
                "torch_rms_norm": torch_rms_norm,
                "qwen_norm": qwen_norm,
                "flashinfer_norm": flashinfer_norm,
                "torch_layer_norm": torch_layer_norm,
            }
        )

    # === 第二部分：精度对比 ===
    print("\n" + "=" * 80)
    print("📊 Accuracy Comparison (vs PyTorch RMSNorm)")
    print("=" * 80)

    for result in all_results:
        config_str = result["config"]
        x = result["x"]
        torch_rms_norm = result["torch_rms_norm"]

        models_to_compare = {
            "Qwen2 RMSNorm": result["qwen_norm"],
            "PyTorch LayerNorm": result["torch_layer_norm"],
        }

        if FLASHINFER_AVAILABLE and result["flashinfer_norm"] is not None:
            models_to_compare["FlashInfer RMSNorm"] = result["flashinfer_norm"]

        compare_outputs(torch_rms_norm, models_to_compare, x, config_str)


if __name__ == "__main__":
    run_benchmark()
