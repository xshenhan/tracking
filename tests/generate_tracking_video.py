#!/usr/bin/env python3
"""
生成完整的追踪可视化视频
"""

import sys
import os
sys.path.insert(0, '/home/xshan/tracking/SpaTrackerV2')

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.SpaTrackV2.models.predictor import Predictor

INPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"
OUTPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"

MODEL_INPUT_SIZE = 384


def load_data():
    video_path = os.path.join(INPUT_DIR, "recorded_video.npz")
    point_path = os.path.join(INPUT_DIR, "query_point.txt")

    data = np.load(video_path)
    rgb_frames = data['rgb']
    depth_frames = data['depth']
    intrinsic = data['intrinsic']

    with open(point_path, 'r') as f:
        line = f.readline().strip()
        x, y = map(int, line.split(','))
        query_point = (x, y)

    return rgb_frames, depth_frames, intrinsic, query_point


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

    track_2d = track2d_pred[..., :2].cpu().numpy()
    track_2d /= scale
    visibility = vis_pred.cpu().numpy()

    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)

    return track_2d[:, 0, :], visibility[:, 0]


def main():
    print("生成追踪可视化视频...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    print("[1] 加载数据...")
    rgb_frames, depth_frames, intrinsic, query_point = load_data()
    num_frames = len(rgb_frames)
    print(f"    帧数: {num_frames}")

    # 加载模型
    print("[2] 加载模型...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()

    # 预处理
    print("[3] 预处理...")
    video_tensor, depth_tensor, intrs, extrs, scale = preprocess(
        rgb_frames, depth_frames, intrinsic, device
    )

    # 追踪
    print("[4] 运行追踪...")
    track_2d, visibility = run_tracking(
        model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device
    )

    # 生成视频
    print("[5] 生成视频...")
    output_path = os.path.join(OUTPUT_DIR, "tracking_result.mp4")

    h, w = rgb_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    # 轨迹历史
    trajectory = []

    for i in range(num_frames):
        frame = rgb_frames[i].copy()
        u, v = track_2d[i]
        vis = visibility[i]

        # 添加到轨迹
        if vis > 0.5:
            trajectory.append((int(u), int(v)))

        # 绘制轨迹（最近50帧）
        recent_traj = trajectory[-50:]
        for j in range(1, len(recent_traj)):
            alpha = j / len(recent_traj)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, recent_traj[j-1], recent_traj[j], color, 2)

        # 绘制当前点
        if vis > 0.5:
            cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(u), int(v)), 12, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (int(u), int(v)), 10, (0, 0, 255), 2)

        # 信息
        cv2.putText(frame, f"Frame: {i}/{num_frames-1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos: ({u:.1f}, {v:.1f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Vis: {vis:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"    已保存: {output_path}")

    # 同时保存一些关键帧
    print("[6] 保存关键帧...")
    key_frames = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]

    for idx, frame_idx in enumerate(key_frames):
        frame = rgb_frames[frame_idx].copy()
        u, v = track_2d[frame_idx]
        vis = visibility[frame_idx]

        # 绘制轨迹到此帧
        traj_to_here = []
        for j in range(frame_idx + 1):
            if visibility[j] > 0.5:
                traj_to_here.append((int(track_2d[j, 0]), int(track_2d[j, 1])))

        for j in range(1, len(traj_to_here)):
            cv2.line(frame, traj_to_here[j-1], traj_to_here[j], (0, 255, 255), 2)

        if vis > 0.5:
            cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(u), int(v)), 12, (255, 255, 255), 2)

        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        save_path = os.path.join(OUTPUT_DIR, f"keyframe_{idx+1}_f{frame_idx:03d}.png")
        cv2.imwrite(save_path, frame)
        print(f"    - keyframe_{idx+1}_f{frame_idx:03d}.png")

    # 清理
    del model
    torch.cuda.empty_cache()

    print("\n完成!")
    print(f"视频路径: {output_path}")


if __name__ == "__main__":
    main()
