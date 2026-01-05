#!/usr/bin/env python3
"""
交互式 PointTracker 测试脚本

操作说明：
- 鼠标点击：设置追踪点
- 空格键：触发追踪，显示当前位置
- r：重置追踪点和缓冲区
- q：退出

运行：
    conda activate SpaTrack
    python interactive_tracker.py
"""

# ============================================
# 重要：必须在导入其他库之前初始化 OpenCV GUI
# 避免 decord/OpenCV 冲突导致 SIGSEGV
# ============================================
import cv2
cv2.namedWindow("_init_", cv2.WINDOW_NORMAL)
cv2.destroyAllWindows()

import sys
import os
sys.path.insert(0, '/home/xshan/tracking')

import numpy as np
import pyrealsense2 as rs
import time

from tracker_tool import PointTracker


class InteractiveTracker:
    def __init__(self, sample_frames=8):
        # RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Tracker
        self.tracker = PointTracker(sample_frames=sample_frames)

        # 状态
        self.query_point = None
        self.last_result = None
        self.show_result = False
        self.result_display_time = 0

        # 当前帧数据（用于点击时立即添加）
        self.current_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.query_point = (x, y)
            self.tracker.reset()
            self.tracker.set_query_point(x, y)

            # 立即添加当前显示的帧作为 first_frame
            if self.current_frame is not None:
                rgb, depth, K = self.current_frame
                self.tracker.add_frame(rgb, depth, K)

            self.last_result = None
            self.show_result = False
            print(f"追踪点已设置: ({x}, {y})")

    def run(self):
        # 启动 RealSense
        profile = self.pipeline.start(self.config)

        # 获取内参
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 对齐
        self.align = rs.align(rs.stream.color)

        # 窗口
        cv2.namedWindow("Interactive Tracker")
        cv2.setMouseCallback("Interactive Tracker", self.mouse_callback)

        print("\n" + "=" * 50)
        print("交互式 PointTracker 测试")
        print("=" * 50)
        print("\n操作说明:")
        print("  鼠标点击 - 设置追踪点")
        print("  空格键   - 触发追踪，显示结果")
        print("  r        - 重置")
        print("  q        - 退出")
        print("=" * 50 + "\n")

        # 跳过前几帧
        for _ in range(30):
            self.pipeline.wait_for_frames()

        frame_count = 0
        fps_time = time.time()
        fps = 0

        try:
            while True:
                # 获取帧
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_meters = depth_image.astype(np.float32) * self.depth_scale

                # 保存当前帧（用于点击时立即添加）
                self.current_frame = (color_image.copy(), depth_meters.copy(), self.intrinsic.copy())

                # 添加帧到 tracker（如果已设置追踪点）
                if self.query_point is not None:
                    self.tracker.add_frame(color_image, depth_meters, self.intrinsic)

                # 计算 FPS
                frame_count += 1
                if time.time() - fps_time > 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                # 显示
                display = color_image.copy()

                # 绘制追踪点（红色）
                if self.query_point is not None:
                    x, y = self.query_point
                    cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                    cv2.circle(display, (x, y), 8, (255, 255, 255), 2)

                # 绘制追踪结果（绿色）
                if self.show_result and self.last_result is not None:
                    u, v = self.last_result['position_2d']
                    vis = self.last_result['visibility']

                    # 绿色追踪点
                    cv2.circle(display, (int(u), int(v)), 10, (0, 255, 0), -1)
                    cv2.circle(display, (int(u), int(v)), 12, (255, 255, 255), 2)

                    # 连线
                    if self.query_point:
                        cv2.line(display, self.query_point, (int(u), int(v)), (0, 255, 255), 2)

                    # 信息
                    pos_3d = self.last_result['position_3d']
                    info_y = 120
                    cv2.putText(display, f"2D: ({u:.1f}, {v:.1f})", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if pos_3d:
                        cv2.putText(display, f"3D: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})m",
                                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Vis: {vis:.2f}", (10, info_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, f"Time: {self.last_result['inference_time']:.3f}s",
                                (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 结果显示 3 秒后自动隐藏
                    if time.time() - self.result_display_time > 3.0:
                        self.show_result = False

                # 状态栏
                status = self.tracker.get_status()
                cv2.putText(display, f"FPS: {fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"Buffer: {status['frame_count']} frames", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if self.query_point is None:
                    cv2.putText(display, "Click to set tracking point", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                else:
                    cv2.putText(display, "Press SPACE to track", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # 底部提示
                cv2.putText(display, "SPACE:track | r:reset | q:quit", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Interactive Tracker", display)

                # 键盘
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):  # 空格
                    if self.query_point is not None and status['frame_count'] > 0:
                        print(f"\n触发追踪... (buffer: {status['frame_count']} frames)")
                        result = self.tracker.get_position()
                        if result:
                            self.last_result = result
                            self.show_result = True
                            self.result_display_time = time.time()
                            print(f"  2D: ({result['position_2d'][0]:.1f}, {result['position_2d'][1]:.1f})")
                            if result['position_3d']:
                                print(f"  3D: ({result['position_3d'][0]:.3f}, {result['position_3d'][1]:.3f}, {result['position_3d'][2]:.3f}) m")
                            print(f"  Visibility: {result['visibility']:.3f}")
                            print(f"  Inference time: {result['inference_time']:.3f}s")
                        else:
                            print("  追踪失败!")
                    else:
                        print("请先点击设置追踪点，并等待录制一些帧")
                elif key == ord('r'):
                    self.query_point = None
                    self.tracker.reset()
                    self.last_result = None
                    self.show_result = False
                    print("已重置")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-frames', type=int, default=8,
                        help='降采样帧数 (default: 8)')
    args = parser.parse_args()

    tracker = InteractiveTracker(sample_frames=args.sample_frames)
    tracker.run()
