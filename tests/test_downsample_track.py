#!/usr/bin/env python3
"""
测试降采样追踪策略
验证：从长视频中降采样 8 帧进行追踪，最后一帧精度是否足够
"""

import sys
import os
sys.path.insert(0, '/home/xshan/tracking/SpaTrackerV2')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time

from models.SpaTrackV2.models.predictor import Predictor

INPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"
ANNO_DIR = "/home/xshan/tracking/tests/outputs/realtime/annotations"

MODEL_INPUT_SIZE = 384
SAMPLE_FRAMES = 8  # 降采样到 8 帧


def load_data():
    video_path = os.path.join(INPUT_DIR, "recorded_video.npz")
    point_path = os.path.join(INPUT_DIR, "query_point.txt")
    anno_path = os.path.join(ANNO_DIR, "ground_truth.txt")

    data = np.load(video_path)
    rgb_frames = data['rgb']
    depth_frames = data['depth']
    intrinsic = data['intrinsic']

    with open(point_path, 'r') as f:
        x, y = map(int, f.readline().strip().split(','))
        query_point = (x, y)

    # 加载 ground truth
    annotations = {}
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            frame_idx = int(parts[0])
            annotations[frame_idx] = (int(parts[1]), int(parts[2]))

    return rgb_frames, depth_frames, intrinsic, query_point, annotations


def downsample_frames(rgb_frames, depth_frames, target_count=8):
    """
    降采样帧：保留第一帧和最后一帧，中间均匀采样
    """
    total = len(rgb_frames)
    if total <= target_count:
        return rgb_frames, depth_frames, list(range(total))

    # 第一帧 + 中间采样 + 最后一帧
    indices = [0]
    step = (total - 1) / (target_count - 1)
    for i in range(1, target_count - 1):
        indices.append(int(i * step))
    indices.append(total - 1)

    sampled_rgb = rgb_frames[indices]
    sampled_depth = depth_frames[indices]

    return sampled_rgb, sampled_depth, indices


def preprocess(frames_bgr, depths, intrinsic_matrix, device):
    num_frames = len(frames_bgr)
    video = frames_bgr[..., ::-1].copy()
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(device).float()
    depth_tensor = depths.copy()

    h, w = video_tensor.shape[2:]
    scale = min(MODEL_INPUT_SIZE / h, MODEL_INPUT_SIZE / w)

    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = F.interpolate(video_tensor, size=(new_h, new_w),
                                     mode='bilinear', align_corners=False)
        depth_tensor = torch.from_numpy(depth_tensor).unsqueeze(1).to(device)
        depth_tensor = F.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest').squeeze(1)
    else:
        scale = 1.0
        depth_tensor = torch.from_numpy(depth_tensor).to(device).float()

    K_scaled = intrinsic_matrix.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale
    intrs = np.tile(K_scaled[None], (num_frames, 1, 1))
    extrs = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

    return video_tensor, depth_tensor, intrs, extrs, scale


@torch.no_grad()
def run_tracking(model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device):
    intrs_t = torch.from_numpy(intrs).to(device)
    extrs_t = torch.from_numpy(extrs).to(device)

    x, y = query_point
    queries = np.array([[0, x * scale, y * scale]], dtype=np.float32)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (_, _, _, _, _, track2d_pred, vis_pred, _, _) = model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs_t,
            extrs=extrs_t,
            queries=queries,
            fps=30,
            full_point=False,
            iters_track=2,
            query_no_BA=True,
            fixed_cam=True,
            stage=1,
            unc_metric=None,
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.0
        )

    track_2d = track2d_pred[..., :2].cpu().numpy() / scale
    visibility = vis_pred.cpu().numpy()
    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)

    return track_2d[:, 0, :], visibility[:, 0]


def main():
    print("=" * 50)
    print("降采样追踪策略验证")
    print("=" * 50)

    device = "cuda"

    # 加载数据
    print("\n[1] 加载数据...")
    rgb_frames, depth_frames, intrinsic, query_point, annotations = load_data()
    total_frames = len(rgb_frames)
    print(f"    总帧数: {total_frames}")
    print(f"    追踪点: {query_point}")

    # 加载模型
    print("\n[2] 加载模型...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()

    # 测试不同降采样率
    print("\n[3] 测试不同降采样配置...")
    print()

    test_configs = [
        ("全部帧", total_frames),
        ("16 帧", 16),
        ("8 帧", 8),
        ("6 帧", 6),
        ("4 帧", 4),
    ]

    # 找到最后一帧对应的 ground truth
    last_frame_idx = total_frames - 1
    # 找最接近的标注帧
    closest_anno_idx = min(annotations.keys(), key=lambda x: abs(x - last_frame_idx))
    gt_x, gt_y = annotations[closest_anno_idx]
    print(f"    最后一帧: {last_frame_idx}, 最近标注帧: {closest_anno_idx}")
    print(f"    Ground Truth: ({gt_x}, {gt_y})")
    print()

    results = []

    for name, sample_count in test_configs:
        # 降采样
        sampled_rgb, sampled_depth, indices = downsample_frames(
            rgb_frames, depth_frames, sample_count
        )

        print(f"    {name}: 采样帧索引 = {indices[:3]}...{indices[-3:]}" if len(indices) > 6 else f"    {name}: 采样帧索引 = {indices}")

        # 预处理
        video_tensor, depth_tensor, intrs, extrs, scale = preprocess(
            sampled_rgb, sampled_depth, intrinsic, device
        )

        # 追踪
        start = time.time()
        track_2d, visibility = run_tracking(
            model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device
        )
        elapsed = time.time() - start

        # 最后一帧结果
        pred_x, pred_y = track_2d[-1]
        vis = visibility[-1]

        # 计算误差（与最近标注帧对比）
        error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        status = "✅" if error < 5.0 else "❌"
        print(f"        推理时间: {elapsed:.3f}s")
        print(f"        预测: ({pred_x:.1f}, {pred_y:.1f}), 误差: {error:.2f}px {status}")
        print()

        results.append({
            'name': name,
            'sample_count': sample_count,
            'time': elapsed,
            'error': error,
            'pred': (pred_x, pred_y),
            'vis': vis
        })

    # 汇总
    print("=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"{'配置':<12} {'帧数':<6} {'时间':<10} {'误差':<10} {'状态':<6}")
    print("-" * 50)
    for r in results:
        status = "✅" if r['error'] < 5.0 else "❌"
        print(f"{r['name']:<12} {r['sample_count']:<6} {r['time']:.3f}s     {r['error']:.2f}px    {status}")

    # 推荐配置
    print()
    valid_results = [r for r in results if r['error'] < 5.0 and r['time'] < 2.0]
    if valid_results:
        best = min(valid_results, key=lambda x: x['time'])
        print(f"推荐配置: {best['name']} (时间: {best['time']:.3f}s, 误差: {best['error']:.2f}px)")

    # 清理
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
