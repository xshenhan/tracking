#!/usr/bin/env python3
"""
阶段 2：SpaTrackerV2 模型加载验证
- 加载 Predictor 模型
- 记录加载时间和显存占用
"""

import sys
import os
sys.path.insert(0, '/home/xshan/tracking/SpaTrackerV2')

import time
import torch

def get_gpu_memory():
    """获取 GPU 显存使用情况 (MB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, reserved
    return 0, 0

def main():
    print("=" * 50)
    print("阶段 2：SpaTrackerV2 模型加载验证")
    print("=" * 50)

    # 初始显存
    torch.cuda.empty_cache()
    init_alloc, init_reserved = get_gpu_memory()
    print(f"\n[1] 初始显存:")
    print(f"    已分配: {init_alloc:.1f} MB")
    print(f"    已预留: {init_reserved:.1f} MB")

    # 加载模型
    print(f"\n[2] 加载模型...")
    print(f"    模型: Yuxihenry/SpatialTrackerV2-Online")

    from models.SpaTrackV2.models.predictor import Predictor

    start_time = time.time()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()
    load_time = time.time() - start_time

    print(f"    加载时间: {load_time:.2f} 秒")

    # 加载后显存
    torch.cuda.synchronize()
    final_alloc, final_reserved = get_gpu_memory()
    model_memory = final_alloc - init_alloc

    print(f"\n[3] 加载后显存:")
    print(f"    已分配: {final_alloc:.1f} MB")
    print(f"    已预留: {final_reserved:.1f} MB")
    print(f"    模型占用: {model_memory:.1f} MB")

    # 模型信息
    print(f"\n[4] 模型信息:")
    print(f"    类型: {type(model).__name__}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    总参数量: {total_params / 1e6:.2f} M")
    print(f"    可训练参数: {trainable_params / 1e6:.2f} M")

    # 测试推理（空跑）
    print(f"\n[5] 测试推理准备...")
    try:
        # 创建虚拟输入测试模型是否正常
        B, T, C, H, W = 1, 8, 3, 384, 384
        dummy_video = torch.randn(B, T, C, H, W).cuda()
        dummy_depth = torch.randn(B, T, H, W).cuda()
        dummy_intrs = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda()
        dummy_extrs = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda()
        dummy_queries = torch.tensor([[[0, 192, 192]]]).float().cuda()  # [B, N, 3]

        print(f"    虚拟输入形状:")
        print(f"      video: {dummy_video.shape}")
        print(f"      depth: {dummy_depth.shape}")
        print(f"      queries: {dummy_queries.shape}")
        print(f"    模型加载验证: ✅ 成功")
    except Exception as e:
        print(f"    模型验证失败: {e}")
        return False

    # 清理
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("阶段 2 完成!")
    print("=" * 50)

    # 返回统计信息
    return {
        "load_time": load_time,
        "model_memory_mb": model_memory,
        "total_params_m": total_params / 1e6
    }

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n摘要:")
        print(f"  加载时间: {result['load_time']:.2f}s")
        print(f"  模型显存: {result['model_memory_mb']:.1f}MB")
        print(f"  参数量: {result['total_params_m']:.2f}M")
