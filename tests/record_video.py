#!/usr/bin/env python3
"""
阶段 5 - 步骤 1：录制视频和追踪点

操作说明：
1. 运行脚本后会显示 RealSense 实时画面
2. 用鼠标点击选择一个追踪点（会显示红色圆点）
3. 按 's' 开始录制（录制时会显示红色 REC 标志）
4. 移动物体，让追踪点跟着移动
5. 按 's' 停止录制
6. 按 'q' 退出并保存

输出文件：
- tests/outputs/realtime/recorded_video.npz  (RGB + Depth + 内参)
- tests/outputs/realtime/query_point.txt     (追踪点坐标)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os

OUTPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"

class VideoRecorder:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.query_point = None
        self.is_recording = False
        self.frames_rgb = []
        self.frames_depth = []
        self.intrinsic_matrix = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.query_point = (x, y)
            print(f"追踪点已设置: ({x}, {y})")

    def start(self):
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 启动 RealSense
        profile = self.pipeline.start(self.config)

        # 获取内参
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 获取深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 对齐
        self.align = rs.align(rs.stream.color)

        # 创建窗口
        cv2.namedWindow("Video Recorder")
        cv2.setMouseCallback("Video Recorder", self.mouse_callback)

        print("\n" + "=" * 50)
        print("视频录制器")
        print("=" * 50)
        print("\n操作说明:")
        print("  1. 鼠标点击选择追踪点")
        print("  2. 按 's' 开始/停止录制")
        print("  3. 按 'r' 重置追踪点")
        print("  4. 按 'q' 退出并保存")
        print("\n提示: 先选择追踪点，再开始录制，然后移动物体")
        print("=" * 50 + "\n")

        # 跳过前几帧让相机稳定
        for _ in range(30):
            self.pipeline.wait_for_frames()

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

                # 录制
                if self.is_recording:
                    self.frames_rgb.append(color_image.copy())
                    self.frames_depth.append(depth_meters.copy())

                # 显示
                display = color_image.copy()

                # 绘制追踪点
                if self.query_point:
                    x, y = self.query_point
                    cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
                    cv2.circle(display, (x, y), 10, (255, 255, 255), 2)

                    # 显示深度
                    d = depth_meters[y, x]
                    cv2.putText(display, f"z={d:.2f}m", (x + 15, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 录制状态
                if self.is_recording:
                    cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (50, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display, f"Frames: {len(self.frames_rgb)}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 状态提示
                status = "Recording..." if self.is_recording else "Ready"
                point_str = f"Point: {self.query_point}" if self.query_point else "Point: None (click to set)"
                cv2.putText(display, status, (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, point_str, (200, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Video Recorder", display)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not self.query_point:
                        print("请先点击设置追踪点!")
                    else:
                        self.is_recording = not self.is_recording
                        if self.is_recording:
                            print("开始录制...")
                            self.frames_rgb = []
                            self.frames_depth = []
                        else:
                            print(f"停止录制，共 {len(self.frames_rgb)} 帧")
                elif key == ord('r'):
                    self.query_point = None
                    print("追踪点已重置")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

        # 保存
        self.save()

    def save(self):
        if not self.frames_rgb:
            print("没有录制的帧，不保存")
            return

        if not self.query_point:
            print("没有设置追踪点，不保存")
            return

        print(f"\n保存数据...")

        # 保存视频数据
        video_path = os.path.join(OUTPUT_DIR, "recorded_video.npz")
        np.savez_compressed(
            video_path,
            rgb=np.array(self.frames_rgb),
            depth=np.array(self.frames_depth),
            intrinsic=self.intrinsic_matrix
        )
        print(f"  - {video_path}")
        print(f"    RGB: {len(self.frames_rgb)} frames, shape: {self.frames_rgb[0].shape}")
        print(f"    Depth: {len(self.frames_depth)} frames")

        # 保存追踪点
        point_path = os.path.join(OUTPUT_DIR, "query_point.txt")
        with open(point_path, 'w') as f:
            f.write(f"{self.query_point[0]},{self.query_point[1]}\n")
        print(f"  - {point_path}")
        print(f"    Point: {self.query_point}")

        # 保存第一帧作为参考
        first_frame_path = os.path.join(OUTPUT_DIR, "first_frame.png")
        first_frame = self.frames_rgb[0].copy()
        x, y = self.query_point
        cv2.circle(first_frame, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(first_frame, (x, y), 10, (255, 255, 255), 2)
        cv2.imwrite(first_frame_path, first_frame)
        print(f"  - {first_frame_path}")

        print("\n录制完成! 现在可以运行 test_realtime_track.py 进行追踪测试")


if __name__ == "__main__":
    recorder = VideoRecorder()
    recorder.start()
