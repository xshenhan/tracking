#!/usr/bin/env python3
"""
阶段 5 - 步骤 2：测试实时追踪效果

前置条件：先运行 record_video.py 录制视频

功能：
1. 加载录制的视频和追踪点
2. 运行 SpaTrackerV2 追踪
3. 每秒保存一帧结果图像
4. 输出追踪性能报告
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
OUTPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"

MODEL_INPUT_SIZE = 384


def load_recorded_data():
    """加载录制的数据"""
    video_path = os.path.join(INPUT_DIR, "recorded_video.npz")
    point_path = os.path.join(INPUT_DIR, "query_point.txt")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到录制的视频: {video_path}\n请先运行 record_video.py")

    if not os.path.exists(point_path):
        raise FileNotFoundError(f"找不到追踪点: {point_path}\n请先运行 record_video.py")

    # 加载视频
    data = np.load(video_path)
    rgb_frames = data['rgb']  # (T, H, W, 3)
    depth_frames = data['depth']  # (T, H, W)
    intrinsic = data['intrinsic']  # (3, 3)

    # 加载追踪点
    with open(point_path, 'r') as f:
        line = f.readline().strip()
        x, y = map(int, line.split(','))
        query_point = (x, y)

    return rgb_frames, depth_frames, intrinsic, query_point


def preprocess(frames_bgr, depths, intrinsic_matrix, device):
    """预处理帧序列"""
    num_frames = len(frames_bgr)

    # BGR -> RGB -> Tensor
    video = frames_bgr[..., ::-1].copy()
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(device)
    video_tensor = video_tensor.float()

    # Depth
    depth_tensor = depths.copy()

    # 缩放
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
def run_tracking(model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device):
    """运行追踪"""
    intrs_t = torch.from_numpy(intrs).to(device)
    extrs_t = torch.from_numpy(extrs).to(device)

    # 构建查询
    x, y = query_point
    queries = np.array([[0, x * scale, y * scale]], dtype=np.float32)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video_out
        ) = model.forward(
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

    # 转换回原始坐标
    track_2d = track2d_pred[..., :2].cpu().numpy()
    track_2d /= scale
    visibility = vis_pred.cpu().numpy()

    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)

    return track_2d, visibility


def pixel_to_3d(u, v, depth, K):
    """2D -> 3D"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_int, u_int = int(round(v)), int(round(u))
    v_int = max(0, min(v_int, depth.shape[0] - 1))
    u_int = max(0, min(u_int, depth.shape[1] - 1))

    z = depth[v_int, u_int]
    if z <= 0:
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def main():
    print("=" * 50)
    print("阶段 5：实时视频追踪验证")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    print("\n[1] 加载录制的数据...")
    try:
        rgb_frames, depth_frames, intrinsic, query_point = load_recorded_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return False

    num_frames = len(rgb_frames)
    print(f"    帧数: {num_frames}")
    print(f"    分辨率: {rgb_frames[0].shape[:2]}")
    print(f"    追踪点: {query_point}")

    # 加载模型
    print("\n[2] 加载模型...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()

    # 预处理
    print("\n[3] 预处理...")
    video_tensor, depth_tensor, intrs, extrs, scale = preprocess(
        rgb_frames, depth_frames, intrinsic, device
    )
    print(f"    Video tensor: {video_tensor.shape}")
    print(f"    Scale: {scale:.4f}")

    # 追踪
    print("\n[4] 运行追踪...")
    start_time = time.time()
    track_2d, visibility = run_tracking(
        model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device
    )
    elapsed = time.time() - start_time

    print(f"    追踪时间: {elapsed:.3f}s")
    print(f"    帧率: {num_frames / elapsed:.1f} FPS")
    print(f"    Track shape: {track_2d.shape}")
    print(f"    Visibility shape: {visibility.shape}")

    # 分析结果
    print("\n[5] 分析结果...")
    track_2d = track_2d[:, 0, :]  # (T, 2)
    visibility = visibility[:, 0]  # (T,)

    # 计算 3D 坐标
    positions_3d = []
    for i in range(num_frames):
        u, v = track_2d[i]
        pos = pixel_to_3d(u, v, depth_frames[i], intrinsic)
        positions_3d.append(pos)

    # 保存每秒一帧
    print("\n[6] 保存结果图像...")
    fps = 30
    save_indices = list(range(0, num_frames, fps))  # 每秒一帧
    if num_frames - 1 not in save_indices:
        save_indices.append(num_frames - 1)

    log_lines = []
    log_lines.append("实时追踪结果\n")
    log_lines.append("=" * 40 + "\n")
    log_lines.append(f"总帧数: {num_frames}\n")
    log_lines.append(f"追踪时间: {elapsed:.3f}s\n")
    log_lines.append(f"帧率: {num_frames / elapsed:.1f} FPS\n\n")

    for idx, frame_idx in enumerate(save_indices):
        frame = rgb_frames[frame_idx].copy()
        u, v = track_2d[frame_idx]
        vis = visibility[frame_idx]
        pos_3d = positions_3d[frame_idx]

        # 绘制追踪点
        if vis > 0.5:
            cv2.circle(frame, (int(u), int(v)), 8, (0, 255, 0), -1)
            cv2.circle(frame, (int(u), int(v)), 10, (255, 255, 255), 2)

            # 标注
            if pos_3d is not None:
                cv2.putText(frame, f"z={pos_3d[2]:.2f}m", (int(u) + 12, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"v={vis:.2f}", (int(u) + 12, int(v) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 帧信息
        cv2.putText(frame, f"Frame {frame_idx}/{num_frames-1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存
        save_path = os.path.join(OUTPUT_DIR, f"frame_{idx+1:03d}.png")
        cv2.imwrite(save_path, frame)
        print(f"    - frame_{idx+1:03d}.png (frame {frame_idx})")

        # 日志
        log_lines.append(f"Frame {frame_idx}:\n")
        log_lines.append(f"  2D: ({u:.1f}, {v:.1f})\n")
        log_lines.append(f"  Visibility: {vis:.3f}\n")
        if pos_3d is not None:
            log_lines.append(f"  3D: ({pos_3d[0]:.4f}, {pos_3d[1]:.4f}, {pos_3d[2]:.4f})\n")
        log_lines.append("\n")

    # 保存日志
    log_path = os.path.join(OUTPUT_DIR, "realtime_log.txt")
    with open(log_path, 'w') as f:
        f.writelines(log_lines)
    print(f"    - realtime_log.txt")

    # 汇总
    print("\n[7] 汇总...")
    valid_vis = visibility[visibility > 0.5]
    print(f"    可见帧比例: {len(valid_vis)}/{num_frames} ({100*len(valid_vis)/num_frames:.1f}%)")
    print(f"    平均可见性: {visibility.mean():.3f}")

    # 计算运动范围
    if len(valid_vis) > 0:
        valid_3d = [p for p in positions_3d if p is not None]
        if valid_3d:
            valid_3d = np.array(valid_3d)
            motion_range = valid_3d.max(axis=0) - valid_3d.min(axis=0)
            print(f"    3D 运动范围: x={motion_range[0]:.3f}m, y={motion_range[1]:.3f}m, z={motion_range[2]:.3f}m")

    # 清理
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("阶段 5 完成!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    main()
