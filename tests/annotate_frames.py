#!/usr/bin/env python3
"""
阶段 5 - 抽帧标注工具

功能：
1. 从录制的视频中均匀抽取 10 帧
2. 显示每帧图像，用户点击标注追踪点的真实位置
3. 保存标注结果供后续对比

操作：
- 鼠标点击标注点的当前位置
- 按 'n' 下一帧
- 按 'r' 重新标注当前帧
- 按 'q' 完成并保存
"""

import numpy as np
import cv2
import os

INPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime"
OUTPUT_DIR = "/home/xshan/tracking/tests/outputs/realtime/annotations"


class FrameAnnotator:
    def __init__(self):
        self.current_point = None
        self.annotations = {}  # frame_idx -> (x, y)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_point = (x, y)
            print(f"  标注点: ({x}, {y})")

    def run(self):
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 加载视频
        video_path = os.path.join(INPUT_DIR, "recorded_video.npz")
        if not os.path.exists(video_path):
            print(f"错误: 找不到录制的视频 {video_path}")
            return False

        data = np.load(video_path)
        rgb_frames = data['rgb']
        num_frames = len(rgb_frames)

        # 加载初始追踪点
        point_path = os.path.join(INPUT_DIR, "query_point.txt")
        with open(point_path, 'r') as f:
            line = f.readline().strip()
            init_x, init_y = map(int, line.split(','))
        print(f"初始追踪点: ({init_x}, {init_y})")

        # 均匀抽取 10 帧
        if num_frames <= 10:
            frame_indices = list(range(num_frames))
        else:
            frame_indices = np.linspace(0, num_frames - 1, 10, dtype=int).tolist()

        print(f"\n总帧数: {num_frames}")
        print(f"抽取帧: {frame_indices}")

        # 创建窗口
        cv2.namedWindow("Annotate Frame")
        cv2.setMouseCallback("Annotate Frame", self.mouse_callback)

        print("\n" + "=" * 50)
        print("帧标注工具")
        print("=" * 50)
        print("\n操作说明:")
        print("  - 鼠标点击标注追踪点的当前位置")
        print("  - 按 'n' 或 空格 确认并下一帧")
        print("  - 按 'r' 重新标注当前帧")
        print("  - 按 'q' 提前完成")
        print("=" * 50 + "\n")

        idx = 0
        while idx < len(frame_indices):
            frame_idx = frame_indices[idx]
            frame = rgb_frames[frame_idx].copy()

            # 显示帧号
            cv2.putText(frame, f"Frame {frame_idx}/{num_frames-1} ({idx+1}/10)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Click to annotate, 'n' for next, 'q' to finish",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示初始点参考（第一帧）
            if frame_idx == 0:
                cv2.circle(frame, (init_x, init_y), 6, (0, 0, 255), -1)
                cv2.putText(frame, "init", (init_x + 10, init_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 显示当前标注
            if self.current_point:
                x, y = self.current_point
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

            cv2.imshow("Annotate Frame", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # 保存当前帧（如果已标注）
                if self.current_point:
                    self.annotations[frame_idx] = self.current_point
                break
            elif key == ord('n') or key == ord(' '):
                if self.current_point:
                    self.annotations[frame_idx] = self.current_point
                    print(f"  Frame {frame_idx}: ({self.current_point[0]}, {self.current_point[1]})")

                    # 保存标注的帧图像
                    save_frame = rgb_frames[frame_idx].copy()
                    x, y = self.current_point
                    cv2.circle(save_frame, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(save_frame, (x, y), 10, (255, 255, 255), 2)
                    cv2.putText(save_frame, f"Frame {frame_idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    save_path = os.path.join(OUTPUT_DIR, f"annotated_{idx+1:02d}_frame{frame_idx:03d}.png")
                    cv2.imwrite(save_path, save_frame)

                    idx += 1
                    self.current_point = None
                else:
                    print("  请先点击标注!")
            elif key == ord('r'):
                self.current_point = None

        cv2.destroyAllWindows()

        # 保存标注结果
        if self.annotations:
            self.save_annotations(frame_indices)
            return True
        else:
            print("没有标注，不保存")
            return False

    def save_annotations(self, frame_indices):
        """保存标注结果"""
        # 保存为文本文件
        anno_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
        with open(anno_path, 'w') as f:
            f.write("# frame_idx, x, y\n")
            for frame_idx in sorted(self.annotations.keys()):
                x, y = self.annotations[frame_idx]
                f.write(f"{frame_idx},{x},{y}\n")

        print(f"\n标注结果已保存:")
        print(f"  - {anno_path}")
        print(f"  - 标注帧数: {len(self.annotations)}")
        print(f"\n标注完成! 现在可以运行对比脚本验证追踪精度")


if __name__ == "__main__":
    annotator = FrameAnnotator()
    annotator.run()
