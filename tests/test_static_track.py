#!/usr/bin/env python3
"""
阶段 3：静态图像追踪验证
- 读取阶段 1 保存的 RGB 和深度图像
- 复制同一帧 8 次模拟视频序列
- 运行追踪并保存结果
"""

import sys
import os
sys.path.insert(0, '/home/xshan/tracking/SpaTrackerV2')

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.SpaTrackV2.models.predictor import Predictor

OUTPUT_DIR = "/home/xshan/tracking/tests/outputs"

# 配置
MODEL_INPUT_SIZE = 384
NUM_FRAMES = 8

def load_data():
    """加载阶段 1 保存的数据"""
    rgb_path = os.path.join(OUTPUT_DIR, "frame_000_rgb.png")
    depth_path = os.path.join(OUTPUT_DIR, "frame_000_depth_raw.npy")
    intrinsic_path = os.path.join(OUTPUT_DIR, "intrinsic.txt")

    # 加载 RGB
    rgb = cv2.imread(rgb_path)
    print(f"    RGB: {rgb.shape}")

    # 加载深度
    depth = np.load(depth_path)
    print(f"    Depth: {depth.shape}, range: {depth[depth > 0].min():.3f}m - {depth.max():.3f}m")

    # 加载内参
    K = np.array([
        [605.903015, 0.0, 323.908539],
        [0.0, 605.985046, 248.735535],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    print(f"    Intrinsic: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    return rgb, depth, K


def preprocess(frames_bgr, depths, intrinsic_matrix, device):
    """预处理帧序列"""
    num_frames = len(frames_bgr)

    # BGR -> RGB -> Tensor
    video = np.stack(frames_bgr, axis=0)[..., ::-1].copy()
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(device)
    video_tensor = video_tensor.float()

    # Depth
    depth_tensor = np.stack(depths, axis=0)

    # 缩放到模型输入尺寸
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

    # 内参缩放
    K_scaled = intrinsic_matrix.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale
    intrs = np.tile(K_scaled[None], (num_frames, 1, 1))

    # 外参 - 固定相机
    extrs = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

    return video_tensor, depth_tensor, intrs, extrs, scale


@torch.no_grad()
def track(model, video_tensor, depth_tensor, intrs, extrs, queries, scale, device):
    """执行追踪"""
    intrs_t = torch.from_numpy(intrs).to(device)
    extrs_t = torch.from_numpy(extrs).to(device)

    # 缩放查询点
    scaled_queries = queries.copy()
    scaled_queries[:, 1:] *= scale

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video_out
        ) = model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs_t,
            extrs=extrs_t,
            queries=scaled_queries,
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

    # 转换回原始坐标
    track_2d = track2d_pred[..., :2].cpu().numpy()
    track_2d /= scale
    visibility = vis_pred.cpu().numpy()

    # 确保 visibility 是 (T, N)
    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)

    return track_2d, visibility


def main():
    print("=" * 50)
    print("阶段 3：静态图像追踪验证")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    print("\n[1] 加载阶段 1 数据...")
    rgb, depth, K = load_data()

    # 选择追踪点（纸盒的一个角，大约在图像下半部分）
    # 根据阶段 1 图像，纸盒大约在 (320, 380) 附近
    query_points = [
        (320, 350),  # 纸盒中心附近
        (280, 380),  # 纸盒左边
        (360, 380),  # 纸盒右边
    ]
    print(f"\n[2] 追踪点: {query_points}")

    # 创建输入图像可视化
    input_vis = rgb.copy()
    for i, (x, y) in enumerate(query_points):
        cv2.circle(input_vis, (x, y), 8, (0, 0, 255), -1)  # 红色
        cv2.circle(input_vis, (x, y), 10, (255, 255, 255), 2)  # 白边
        cv2.putText(input_vis, str(i), (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "track_input.png"), input_vis)
    print(f"    已保存: track_input.png")

    # 复制帧模拟视频序列
    print(f"\n[3] 创建 {NUM_FRAMES} 帧视频序列...")
    frames = [rgb.copy() for _ in range(NUM_FRAMES)]
    depths = [depth.copy() for _ in range(NUM_FRAMES)]

    # 加载模型
    print("\n[4] 加载模型...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()

    # 预处理
    print("\n[5] 预处理...")
    video_tensor, depth_tensor, intrs, extrs, scale = preprocess(frames, depths, K, device)
    print(f"    Video tensor: {video_tensor.shape}")
    print(f"    Depth tensor: {depth_tensor.shape}")
    print(f"    Scale: {scale:.4f}")

    # 构建查询
    queries = np.array([[0, x, y] for x, y in query_points], dtype=np.float32)
    print(f"    Queries: {queries.shape}")

    # 追踪
    print("\n[6] 执行追踪...")
    import time
    start = time.time()
    track_2d, visibility = track(model, video_tensor, depth_tensor, intrs, extrs, queries, scale, device)
    elapsed = time.time() - start
    print(f"    追踪时间: {elapsed:.3f}s")
    print(f"    Track 2D shape: {track_2d.shape}")  # (T, N, 2)
    print(f"    Visibility shape: {visibility.shape}")  # (T, N)

    # 分析结果
    print("\n[7] 分析结果...")
    log_lines = []
    log_lines.append("静态图像追踪结果\n")
    log_lines.append("=" * 40 + "\n\n")

    all_stable = True
    for i, (ox, oy) in enumerate(query_points):
        track_i = track_2d[:, i, :]  # (T, 2)
        vis_i = visibility[:, i]  # (T,)

        # 计算偏移
        offsets = np.sqrt((track_i[:, 0] - ox) ** 2 + (track_i[:, 1] - oy) ** 2)
        max_offset = offsets.max()
        mean_offset = offsets.mean()
        mean_vis = vis_i.mean()

        stable = max_offset < 5.0 and mean_vis > 0.8
        status = "✅ 稳定" if stable else "❌ 不稳定"
        if not stable:
            all_stable = False

        log_lines.append(f"点 {i}: 原始({ox}, {oy})\n")
        log_lines.append(f"  最大偏移: {max_offset:.2f} px\n")
        log_lines.append(f"  平均偏移: {mean_offset:.2f} px\n")
        log_lines.append(f"  平均可见性: {mean_vis:.3f}\n")
        log_lines.append(f"  状态: {status}\n\n")

        print(f"    点 {i}: 最大偏移={max_offset:.2f}px, 平均可见性={mean_vis:.3f} {status}")

    # 保存日志
    with open(os.path.join(OUTPUT_DIR, "track_log.txt"), 'w') as f:
        f.writelines(log_lines)
    print(f"    已保存: track_log.txt")

    # 创建结果可视化（最后一帧）
    result_vis = rgb.copy()
    last_frame_idx = NUM_FRAMES - 1

    for i, (ox, oy) in enumerate(query_points):
        tx, ty = track_2d[last_frame_idx, i, :]
        vis = visibility[last_frame_idx, i]

        # 原始点（红色）
        cv2.circle(result_vis, (int(ox), int(oy)), 6, (0, 0, 255), -1)

        # 追踪点（绿色，如果可见）
        if vis > 0.5:
            cv2.circle(result_vis, (int(tx), int(ty)), 8, (0, 255, 0), -1)
            cv2.circle(result_vis, (int(tx), int(ty)), 10, (255, 255, 255), 2)

        # 标注
        cv2.putText(result_vis, f"{i}:v={vis:.2f}", (int(tx) + 12, int(ty) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "track_result.png"), result_vis)
    print(f"    已保存: track_result.png")

    # 清理
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    if all_stable:
        print("阶段 3 完成! ✅ 所有追踪点稳定")
    else:
        print("阶段 3 完成! ⚠️ 部分追踪点不稳定")
    print("=" * 50)

    return all_stable


if __name__ == "__main__":
    main()
