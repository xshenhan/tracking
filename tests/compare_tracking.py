#!/usr/bin/env python3
"""
阶段 5 - 追踪结果对比验证

功能：
1. 运行 SpaTrackerV2 追踪
2. 加载用户标注的 ground truth
3. 对比追踪结果和标注结果
4. 计算误差，判断是否 < 5 像素
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
OUTPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"

MODEL_INPUT_SIZE = 384
ERROR_THRESHOLD = 5.0  # 像素


def load_data():
    """加载数据"""
    # 加载视频
    video_path = os.path.join(INPUT_DIR, "recorded_video.npz")
    data = np.load(video_path)
    rgb_frames = data['rgb']
    depth_frames = data['depth']
    intrinsic = data['intrinsic']

    # 加载追踪点
    point_path = os.path.join(INPUT_DIR, "query_point.txt")
    with open(point_path, 'r') as f:
        line = f.readline().strip()
        x, y = map(int, line.split(','))
        query_point = (x, y)

    # 加载标注
    anno_path = os.path.join(ANNO_DIR, "ground_truth.txt")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"找不到标注文件: {anno_path}\n请先运行 annotate_frames.py")

    annotations = {}
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            frame_idx = int(parts[0])
            x, y = int(parts[1]), int(parts[2])
            annotations[frame_idx] = (x, y)

    return rgb_frames, depth_frames, intrinsic, query_point, annotations


def preprocess(frames_bgr, depths, intrinsic_matrix, device):
    """预处理"""
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
    """运行追踪"""
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

    return track_2d[:, 0, :], visibility[:, 0]  # (T, 2), (T,)


def main():
    print("=" * 50)
    print("阶段 5：追踪结果对比验证")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    print("\n[1] 加载数据...")
    rgb_frames, depth_frames, intrinsic, query_point, annotations = load_data()
    num_frames = len(rgb_frames)
    print(f"    帧数: {num_frames}")
    print(f"    追踪点: {query_point}")
    print(f"    标注帧数: {len(annotations)}")

    # 加载模型
    print("\n[2] 加载模型...")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model = model.cuda().eval()

    # 预处理
    print("\n[3] 预处理...")
    video_tensor, depth_tensor, intrs, extrs, scale = preprocess(
        rgb_frames, depth_frames, intrinsic, device
    )

    # 追踪
    print("\n[4] 运行追踪...")
    start_time = time.time()
    track_2d, visibility = run_tracking(
        model, video_tensor, depth_tensor, intrs, extrs, query_point, scale, device
    )
    elapsed = time.time() - start_time
    print(f"    追踪时间: {elapsed:.3f}s")
    print(f"    帧率: {num_frames / elapsed:.1f} FPS")

    # 对比
    print("\n[5] 对比结果...")
    print(f"    误差阈值: {ERROR_THRESHOLD} 像素")
    print()

    results = []
    log_lines = []
    log_lines.append("追踪结果对比\n")
    log_lines.append("=" * 50 + "\n")
    log_lines.append(f"误差阈值: {ERROR_THRESHOLD} 像素\n\n")

    for frame_idx in sorted(annotations.keys()):
        gt_x, gt_y = annotations[frame_idx]
        pred_x, pred_y = track_2d[frame_idx]
        vis = visibility[frame_idx]

        error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        passed = error < ERROR_THRESHOLD

        status = "✅ OK" if passed else "❌ FAIL"
        print(f"    Frame {frame_idx:3d}: GT=({gt_x:3d},{gt_y:3d}) Pred=({pred_x:6.1f},{pred_y:6.1f}) "
              f"Error={error:5.2f}px Vis={vis:.2f} {status}")

        results.append({
            'frame_idx': frame_idx,
            'gt': (gt_x, gt_y),
            'pred': (pred_x, pred_y),
            'error': error,
            'visibility': vis,
            'passed': passed
        })

        log_lines.append(f"Frame {frame_idx}:\n")
        log_lines.append(f"  Ground Truth: ({gt_x}, {gt_y})\n")
        log_lines.append(f"  Prediction:   ({pred_x:.1f}, {pred_y:.1f})\n")
        log_lines.append(f"  Error:        {error:.2f} px\n")
        log_lines.append(f"  Visibility:   {vis:.2f}\n")
        log_lines.append(f"  Status:       {status}\n\n")

    # 统计
    errors = [r['error'] for r in results]
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    pass_rate = passed_count / total_count * 100

    print()
    print(f"    -------------------")
    print(f"    通过: {passed_count}/{total_count} ({pass_rate:.0f}%)")
    print(f"    平均误差: {np.mean(errors):.2f} px")
    print(f"    最大误差: {np.max(errors):.2f} px")
    print(f"    最小误差: {np.min(errors):.2f} px")

    log_lines.append("=" * 50 + "\n")
    log_lines.append(f"通过: {passed_count}/{total_count} ({pass_rate:.0f}%)\n")
    log_lines.append(f"平均误差: {np.mean(errors):.2f} px\n")
    log_lines.append(f"最大误差: {np.max(errors):.2f} px\n")
    log_lines.append(f"最小误差: {np.min(errors):.2f} px\n")

    # 保存对比图像
    print("\n[6] 保存对比图像...")
    for i, r in enumerate(results):
        frame_idx = r['frame_idx']
        frame = rgb_frames[frame_idx].copy()

        gt_x, gt_y = r['gt']
        pred_x, pred_y = r['pred']

        # 绘制 GT（红色）
        cv2.circle(frame, (gt_x, gt_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, "GT", (gt_x + 10, gt_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 绘制预测（绿色）
        cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (0, 255, 0), 2)
        cv2.putText(frame, "Pred", (int(pred_x) + 10, int(pred_y) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 连线
        cv2.line(frame, (gt_x, gt_y), (int(pred_x), int(pred_y)), (255, 255, 0), 1)

        # 信息
        status = "OK" if r['passed'] else "FAIL"
        cv2.putText(frame, f"Frame {frame_idx} | Error: {r['error']:.2f}px | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        save_path = os.path.join(OUTPUT_DIR, f"compare_{i+1:02d}_frame{frame_idx:03d}.png")
        cv2.imwrite(save_path, frame)
        print(f"    - compare_{i+1:02d}_frame{frame_idx:03d}.png")

    # 保存日志
    log_path = os.path.join(OUTPUT_DIR, "compare_log.txt")
    with open(log_path, 'w') as f:
        f.writelines(log_lines)
    print(f"    - compare_log.txt")

    # 清理
    del model
    torch.cuda.empty_cache()

    # 最终判断
    print("\n" + "=" * 50)
    if pass_rate >= 80:
        print(f"阶段 5 完成! ✅ 追踪精度合格 ({pass_rate:.0f}% 通过)")
    else:
        print(f"阶段 5 完成! ❌ 追踪精度不足 ({pass_rate:.0f}% 通过)")
    print("=" * 50)

    return pass_rate >= 80


if __name__ == "__main__":
    main()
