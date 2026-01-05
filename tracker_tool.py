#!/usr/bin/env python3
"""
PointTracker - 基于 SpaTrackerV2 的点追踪工具

使用方法:
    tracker = PointTracker(sample_frames=8)

    # 设置追踪点（首帧像素坐标）
    tracker.set_query_point(x, y)

    # 持续添加帧
    tracker.add_frame(rgb, depth, intrinsic)

    # 任意时刻获取当前位置
    result = tracker.get_position()
    # result = {
    #     'position_2d': (u, v),
    #     'position_3d': (x, y, z),  # 相机坐标系
    #     'visibility': 0.95,
    #     'frame_count': 100,
    #     'inference_time': 0.105
    # }
"""

import sys
import os
sys.path.insert(0, '/home/xshan/tracking/SpaTrackerV2')

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import threading
import time
from typing import Optional, Tuple, Dict, Any

from models.SpaTrackV2.models.predictor import Predictor


class PointTracker:
    """
    点追踪器：持续录制视频，按需推理返回当前位置

    Args:
        sample_frames: 降采样帧数，用于推理（默认 8）
        buffer_size: 缓冲区大小，保留最近 N 帧（默认 1000，约 33 秒 @ 30fps）
        model_input_size: 模型输入尺寸（默认 384）
        device: 运行设备（默认 cuda）
    """

    def __init__(
        self,
        sample_frames: int = 8,
        buffer_size: int = 1000,
        model_input_size: int = 384,
        device: str = "cuda"
    ):
        self.sample_frames = sample_frames
        self.buffer_size = buffer_size
        self.model_input_size = model_input_size
        self.device = device

        # 缓冲区
        self.frame_buffer: deque = deque(maxlen=buffer_size)  # (rgb, depth, intrinsic)
        self.first_frame: Optional[Tuple] = None  # 保存第一帧（有标注）

        # 追踪点
        self.query_point: Optional[Tuple[int, int]] = None

        # 模型（延迟加载）
        self.model: Optional[Predictor] = None
        self.model_loaded = False

        # 线程锁
        self.lock = threading.Lock()

        # 状态
        self.frame_count = 0
        self.last_result: Optional[Dict] = None

    def _load_model(self):
        """延迟加载模型"""
        if not self.model_loaded:
            print("[PointTracker] Loading model...")
            start = time.time()
            self.model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"[PointTracker] Model loaded in {time.time() - start:.2f}s")

    def set_query_point(self, x: int, y: int):
        """
        设置追踪点（首帧像素坐标）

        Args:
            x: 像素 x 坐标
            y: 像素 y 坐标
        """
        with self.lock:
            self.query_point = (x, y)
            print(f"[PointTracker] Query point set: ({x}, {y})")

    def add_frame(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray):
        """
        添加帧到缓冲区

        Args:
            rgb: RGB 图像 (H, W, 3)，BGR 或 RGB 格式均可
            depth: 深度图 (H, W)，单位：米
            intrinsic: 内参矩阵 (3, 3)
        """
        with self.lock:
            # 保存第一帧（单独保存，不加入 buffer）
            if self.first_frame is None:
                self.first_frame = (rgb.copy(), depth.copy(), intrinsic.copy())
                self.frame_count += 1
                print(f"[PointTracker] First frame saved")
                return  # 第一帧不加入 buffer，避免重复

            # 后续帧添加到缓冲区
            self.frame_buffer.append((rgb.copy(), depth.copy(), intrinsic.copy()))
            self.frame_count += 1

    def _downsample_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从缓冲区降采样帧

        Returns:
            rgb_frames: (N, H, W, 3)
            depth_frames: (N, H, W)
            intrinsic: (3, 3)
        """
        if self.first_frame is None:
            raise ValueError("No first frame set")

        # 采样帧：first_frame（有标注）+ buffer 中的帧
        sampled_rgb = [self.first_frame[0]]
        sampled_depth = [self.first_frame[1]]

        # 如果 buffer 为空，只返回第一帧
        if len(self.frame_buffer) == 0:
            rgb_frames = np.array(sampled_rgb)
            depth_frames = np.array(sampled_depth)
            return rgb_frames, depth_frames, self.first_frame[2]

        # 获取 buffer 中的所有帧
        all_frames = list(self.frame_buffer)
        total = len(all_frames)

        # 从 buffer 中采样 sample_frames - 1 帧（第一帧已经是 first_frame）
        if total <= self.sample_frames - 1:
            # 帧数不足，全部使用
            indices = list(range(total))
        else:
            # 均匀采样，确保包含最后一帧
            indices = []
            step = total / (self.sample_frames - 1)
            for i in range(self.sample_frames - 1):
                indices.append(min(int(i * step), total - 1))
            # 确保最后一帧被包含
            if indices[-1] != total - 1:
                indices[-1] = total - 1

        # 添加采样的帧
        for idx in indices:
            sampled_rgb.append(all_frames[idx][0])
            sampled_depth.append(all_frames[idx][1])

        # 转换为数组
        rgb_frames = np.array(sampled_rgb)
        depth_frames = np.array(sampled_depth)
        intrinsic = self.first_frame[2]

        return rgb_frames, depth_frames, intrinsic

    def _preprocess(
        self,
        frames_bgr: np.ndarray,
        depths: np.ndarray,
        intrinsic_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, float]:
        """预处理帧序列"""
        num_frames = len(frames_bgr)

        # BGR -> RGB -> Tensor
        video = frames_bgr[..., ::-1].copy()
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(self.device).float()
        depth_tensor = depths.copy()

        # 缩放
        h, w = video_tensor.shape[2:]
        scale = min(self.model_input_size / h, self.model_input_size / w)

        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video_tensor = F.interpolate(video_tensor, size=(new_h, new_w),
                                         mode='bilinear', align_corners=False)
            depth_tensor = torch.from_numpy(depth_tensor).unsqueeze(1).to(self.device)
            depth_tensor = F.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest').squeeze(1)
        else:
            scale = 1.0
            depth_tensor = torch.from_numpy(depth_tensor).to(self.device).float()

        # 内参缩放
        K_scaled = intrinsic_matrix.copy()
        K_scaled[0, :] *= scale
        K_scaled[1, :] *= scale
        intrs = np.tile(K_scaled[None], (num_frames, 1, 1))
        extrs = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

        return video_tensor, depth_tensor, intrs, extrs, scale

    @torch.no_grad()
    def _run_tracking(
        self,
        video_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        intrs: np.ndarray,
        extrs: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """运行追踪"""
        intrs_t = torch.from_numpy(intrs).to(self.device)
        extrs_t = torch.from_numpy(extrs).to(self.device)

        x, y = self.query_point
        queries = np.array([[0, x * scale, y * scale]], dtype=np.float32)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (_, _, _, _, _, track2d_pred, vis_pred, _, _) = self.model.forward(
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

    def _pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: np.ndarray,
        K: np.ndarray
    ) -> Optional[np.ndarray]:
        """2D -> 3D 转换"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        v_int = max(0, min(int(round(v)), depth.shape[0] - 1))
        u_int = max(0, min(int(round(u)), depth.shape[1] - 1))

        z = depth[v_int, u_int]
        if z <= 0:
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    def get_position(self, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        获取当前位置（触发推理）

        Args:
            force_reload: 是否强制重新加载模型

        Returns:
            dict: {
                'position_2d': (u, v),           # 像素坐标
                'position_3d': (x, y, z),        # 相机坐标系，单位米
                'visibility': float,             # 可见性分数
                'frame_count': int,              # 当前缓冲区帧数
                'inference_time': float          # 推理时间（秒）
            }
            如果无法追踪返回 None
        """
        # 检查状态
        if self.query_point is None:
            print("[PointTracker] Error: Query point not set")
            return None

        if self.first_frame is None:
            print("[PointTracker] Error: No frames added")
            return None

        # 模型需要至少 4 帧
        total_frames = 1 + len(self.frame_buffer)  # first_frame + buffer
        if total_frames < 4:
            print(f"[PointTracker] Error: Need at least 4 frames for tracking (currently {total_frames})")
            return None

        # 加载模型
        if force_reload:
            self.model_loaded = False
        self._load_model()

        start_time = time.time()

        with self.lock:
            # 降采样
            rgb_frames, depth_frames, intrinsic = self._downsample_frames()

        # 预处理
        video_tensor, depth_tensor, intrs, extrs, scale = self._preprocess(
            rgb_frames, depth_frames, intrinsic
        )

        # 追踪
        track_2d, visibility = self._run_tracking(
            video_tensor, depth_tensor, intrs, extrs, scale
        )

        # 最后一帧结果
        u, v = track_2d[-1]
        vis = visibility[-1]

        # 获取最后一帧的深度图用于 3D 转换
        with self.lock:
            if len(self.frame_buffer) > 0:
                last_depth = list(self.frame_buffer)[-1][1]
            else:
                last_depth = self.first_frame[1]

        # 2D -> 3D
        pos_3d = self._pixel_to_3d(u, v, last_depth, intrinsic)

        inference_time = time.time() - start_time

        result = {
            'position_2d': (float(u), float(v)),
            'position_3d': tuple(pos_3d.tolist()) if pos_3d is not None else None,
            'visibility': float(vis),
            'frame_count': len(self.frame_buffer),
            'inference_time': inference_time
        }

        self.last_result = result
        return result

    def reset(self):
        """重置追踪器状态（保留模型）"""
        with self.lock:
            self.frame_buffer.clear()
            self.first_frame = None
            self.query_point = None
            self.frame_count = 0
            self.last_result = None
        print("[PointTracker] Reset complete")

    def get_status(self) -> Dict[str, Any]:
        """获取追踪器状态"""
        return {
            'model_loaded': self.model_loaded,
            'query_point': self.query_point,
            'frame_count': len(self.frame_buffer),
            'total_frames_added': self.frame_count,
            'sample_frames': self.sample_frames,
            'buffer_size': self.buffer_size,
            'has_first_frame': self.first_frame is not None
        }


# ============== 便捷函数 ==============

def create_tracker(sample_frames: int = 8, **kwargs) -> PointTracker:
    """创建追踪器实例"""
    return PointTracker(sample_frames=sample_frames, **kwargs)


# ============== 测试 ==============

if __name__ == "__main__":
    import cv2

    print("=" * 50)
    print("PointTracker 单元测试")
    print("=" * 50)

    # 加载测试数据
    INPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"

    data = np.load(os.path.join(INPUT_DIR, "recorded_video.npz"))
    rgb_frames = data['rgb']
    depth_frames = data['depth']
    intrinsic = data['intrinsic']

    with open(os.path.join(INPUT_DIR, "query_point.txt"), 'r') as f:
        qx, qy = map(int, f.readline().strip().split(','))

    print(f"\n[1] 创建追踪器...")
    tracker = PointTracker(sample_frames=8)
    print(f"    状态: {tracker.get_status()}")

    print(f"\n[2] 设置追踪点: ({qx}, {qy})")
    tracker.set_query_point(qx, qy)

    print(f"\n[3] 添加帧...")
    for i in range(len(rgb_frames)):
        tracker.add_frame(rgb_frames[i], depth_frames[i], intrinsic)
    print(f"    添加了 {len(rgb_frames)} 帧")
    print(f"    状态: {tracker.get_status()}")

    print(f"\n[4] 获取位置...")
    result = tracker.get_position()

    if result:
        print(f"    2D 位置: ({result['position_2d'][0]:.1f}, {result['position_2d'][1]:.1f})")
        if result['position_3d']:
            print(f"    3D 位置: ({result['position_3d'][0]:.4f}, {result['position_3d'][1]:.4f}, {result['position_3d'][2]:.4f}) m")
        print(f"    可见性: {result['visibility']:.3f}")
        print(f"    推理时间: {result['inference_time']:.3f}s")

    print(f"\n[5] 重置追踪器...")
    tracker.reset()
    print(f"    状态: {tracker.get_status()}")

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
