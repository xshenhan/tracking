#!/usr/bin/env python3
"""
集成测试脚本 - 端到端验证 PointTracker

测试场景：
1. 静止物体 - 物体不动，追踪点应保持稳定
2. 移动物体 - 物体移动，追踪点应跟随
3. 遮挡测试 - 物体被遮挡后恢复，追踪点应恢复

操作说明：
- 鼠标点击：设置追踪点
- 空格键：开始/停止当前测试
- 1/2/3：切换测试场景
- s：保存当前测试结果
- r：重置
- q：退出

运行：
    conda activate SpaTrack
    python tests/test_integration.py
"""

# OpenCV GUI 初始化（必须在其他导入之前）
import cv2
cv2.namedWindow("_init_", cv2.WINDOW_NORMAL)
cv2.destroyAllWindows()

import sys
import os
sys.path.insert(0, '/home/xshan/tracking')

import numpy as np
import pyrealsense2 as rs
import time
from datetime import datetime

from tracker_tool import PointTracker


class IntegrationTest:
    def __init__(self):
        # RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Tracker
        self.tracker = PointTracker(sample_frames=8)

        # 测试状态
        self.test_mode = 1  # 1: 静止, 2: 移动, 3: 遮挡
        self.test_names = {1: "Static", 2: "Moving", 3: "Occlusion"}
        self.is_testing = False
        self.test_start_time = 0
        self.test_results = []

        # 追踪状态
        self.query_point = None
        self.current_frame = None
        self.tracking_history = []  # (time, x, y, vis)

        # 输出目录
        self.output_dir = "/home/xshan/tracking/tests/outputs/integration"
        os.makedirs(self.output_dir, exist_ok=True)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.query_point = (x, y)
            self.tracker.reset()
            self.tracker.set_query_point(x, y)

            # 立即添加当前帧
            if self.current_frame is not None:
                rgb, depth, K = self.current_frame
                self.tracker.add_frame(rgb, depth, K)

            self.tracking_history = []
            print(f"追踪点设置: ({x}, {y})")

    def run_tracking(self):
        """执行一次追踪并记录结果"""
        result = self.tracker.get_position()
        if result:
            elapsed = time.time() - self.test_start_time
            self.tracking_history.append({
                'time': elapsed,
                'x': result['position_2d'][0],
                'y': result['position_2d'][1],
                'vis': result['visibility'],
                'pos_3d': result['position_3d']
            })
            return result
        return None

    def analyze_results(self):
        """分析测试结果"""
        if len(self.tracking_history) < 2:
            return None

        positions = [(h['x'], h['y']) for h in self.tracking_history]
        visibilities = [h['vis'] for h in self.tracking_history]

        # 计算统计
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # 位移统计
        total_displacement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_displacement += np.sqrt(dx*dx + dy*dy)

        # 初始点到最终点的距离
        if self.query_point:
            start_to_end = np.sqrt(
                (positions[-1][0] - self.query_point[0])**2 +
                (positions[-1][1] - self.query_point[1])**2
            )
        else:
            start_to_end = 0

        analysis = {
            'test_mode': self.test_mode,
            'test_name': self.test_names[self.test_mode],
            'num_samples': len(self.tracking_history),
            'duration': self.tracking_history[-1]['time'],
            'x_range': (min(x_coords), max(x_coords)),
            'y_range': (min(y_coords), max(y_coords)),
            'x_std': np.std(x_coords),
            'y_std': np.std(y_coords),
            'total_displacement': total_displacement,
            'start_to_end_distance': start_to_end,
            'mean_visibility': np.mean(visibilities),
            'min_visibility': min(visibilities),
        }

        # 判断测试结果
        if self.test_mode == 1:  # 静止物体
            # 标准差应该很小
            passed = analysis['x_std'] < 5 and analysis['y_std'] < 5
            analysis['criterion'] = f"Stability (std < 5px): x_std={analysis['x_std']:.2f}, y_std={analysis['y_std']:.2f}"
        elif self.test_mode == 2:  # 移动物体
            # 应该有明显位移
            passed = analysis['total_displacement'] > 20
            analysis['criterion'] = f"Tracking motion (displacement > 20px): {analysis['total_displacement']:.1f}px"
        else:  # 遮挡
            # 遮挡时可见性下降，恢复后可见性恢复
            passed = analysis['min_visibility'] < 0.5 and analysis['mean_visibility'] > 0.3
            analysis['criterion'] = f"Occlusion detection (min_vis < 0.5): {analysis['min_visibility']:.2f}"

        analysis['passed'] = passed
        return analysis

    def save_results(self):
        """保存测试结果"""
        if len(self.tracking_history) < 2:
            print("数据不足，无法保存")
            return

        analysis = self.analyze_results()
        if analysis is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.test_names[self.test_mode].lower()

        # 保存轨迹数据
        data_path = os.path.join(self.output_dir, f"{test_name}_{timestamp}.npz")
        np.savez(data_path,
                 query_point=self.query_point,
                 history=self.tracking_history,
                 analysis=analysis)

        # 保存分析报告
        report_path = os.path.join(self.output_dir, f"{test_name}_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Integration Test Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Test: {analysis['test_name']}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Duration: {analysis['duration']:.2f}s\n")
            f.write(f"Samples: {analysis['num_samples']}\n\n")
            f.write(f"Results:\n")
            f.write(f"  X range: {analysis['x_range'][0]:.1f} - {analysis['x_range'][1]:.1f}\n")
            f.write(f"  Y range: {analysis['y_range'][0]:.1f} - {analysis['y_range'][1]:.1f}\n")
            f.write(f"  X std: {analysis['x_std']:.2f} px\n")
            f.write(f"  Y std: {analysis['y_std']:.2f} px\n")
            f.write(f"  Total displacement: {analysis['total_displacement']:.1f} px\n")
            f.write(f"  Mean visibility: {analysis['mean_visibility']:.3f}\n")
            f.write(f"  Min visibility: {analysis['min_visibility']:.3f}\n\n")
            f.write(f"Criterion: {analysis['criterion']}\n")
            f.write(f"Result: {'PASSED' if analysis['passed'] else 'FAILED'}\n")

        status = "✅ PASSED" if analysis['passed'] else "❌ FAILED"
        print(f"\n结果已保存: {report_path}")
        print(f"测试结果: {status}")
        print(f"  {analysis['criterion']}")

        self.test_results.append(analysis)

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
        cv2.namedWindow("Integration Test")
        cv2.setMouseCallback("Integration Test", self.mouse_callback)

        print("\n" + "=" * 50)
        print("PointTracker 集成测试")
        print("=" * 50)
        print("\n测试场景:")
        print("  1 - 静止物体（物体不动，验证稳定性）")
        print("  2 - 移动物体（移动物体，验证跟踪）")
        print("  3 - 遮挡测试（遮挡后恢复，验证鲁棒性）")
        print("\n操作:")
        print("  点击   - 设置追踪点")
        print("  空格   - 开始/停止测试")
        print("  1/2/3  - 切换场景")
        print("  s      - 保存结果")
        print("  r      - 重置")
        print("  q      - 退出")
        print("=" * 50 + "\n")

        # 跳过前几帧
        for _ in range(30):
            self.pipeline.wait_for_frames()

        frame_count = 0
        fps_time = time.time()
        fps = 0
        last_track_time = 0

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

                # 保存当前帧
                self.current_frame = (color_image.copy(), depth_meters.copy(), self.intrinsic.copy())

                # 添加帧到 tracker
                if self.query_point is not None:
                    self.tracker.add_frame(color_image, depth_meters, self.intrinsic)

                # 测试中：定期追踪
                if self.is_testing and self.query_point is not None:
                    status = self.tracker.get_status()
                    if status['frame_count'] >= 4:
                        current_time = time.time()
                        if current_time - last_track_time > 0.5:  # 每 0.5 秒追踪一次
                            self.run_tracking()
                            last_track_time = current_time

                # 计算 FPS
                frame_count += 1
                if time.time() - fps_time > 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                # 显示
                display = color_image.copy()

                # 测试模式标题
                mode_color = (0, 255, 255) if self.is_testing else (200, 200, 200)
                cv2.putText(display, f"Mode {self.test_mode}: {self.test_names[self.test_mode]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

                if self.is_testing:
                    elapsed = time.time() - self.test_start_time
                    cv2.putText(display, f"TESTING... {elapsed:.1f}s",
                                (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 追踪点（红色）
                if self.query_point is not None:
                    x, y = self.query_point
                    cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                    cv2.circle(display, (x, y), 8, (255, 255, 255), 2)

                # 绘制轨迹历史
                if len(self.tracking_history) > 1:
                    points = [(int(h['x']), int(h['y'])) for h in self.tracking_history]
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                        cv2.line(display, points[i-1], points[i], color, 2)

                    # 当前位置（绿色）
                    last = self.tracking_history[-1]
                    cv2.circle(display, (int(last['x']), int(last['y'])), 10, (0, 255, 0), -1)
                    cv2.circle(display, (int(last['x']), int(last['y'])), 12, (255, 255, 255), 2)

                    # 可见性
                    cv2.putText(display, f"Vis: {last['vis']:.2f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 状态栏
                status = self.tracker.get_status()
                cv2.putText(display, f"FPS: {fps} | Buffer: {status['frame_count']}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 提示
                if self.query_point is None:
                    cv2.putText(display, "Click to set tracking point",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                elif not self.is_testing:
                    cv2.putText(display, "Press SPACE to start test",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # 底部控制提示
                cv2.putText(display, "1/2/3:mode | SPACE:test | s:save | r:reset | q:quit",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Integration Test", display)

                # 键盘
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.test_mode = 1
                    print(f"切换到测试 1: 静止物体")
                elif key == ord('2'):
                    self.test_mode = 2
                    print(f"切换到测试 2: 移动物体")
                elif key == ord('3'):
                    self.test_mode = 3
                    print(f"切换到测试 3: 遮挡测试")
                elif key == ord(' '):
                    if self.query_point is not None:
                        self.is_testing = not self.is_testing
                        if self.is_testing:
                            self.test_start_time = time.time()
                            self.tracking_history = []
                            print(f"开始测试: {self.test_names[self.test_mode]}")
                        else:
                            print(f"停止测试")
                    else:
                        print("请先点击设置追踪点")
                elif key == ord('s'):
                    self.save_results()
                elif key == ord('r'):
                    self.query_point = None
                    self.tracker.reset()
                    self.tracking_history = []
                    self.is_testing = False
                    print("已重置")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

            # 打印总结
            if self.test_results:
                print("\n" + "=" * 50)
                print("测试总结")
                print("=" * 50)
                for r in self.test_results:
                    status = "✅" if r['passed'] else "❌"
                    print(f"{status} {r['test_name']}: {r['criterion']}")


if __name__ == "__main__":
    test = IntegrationTest()
    test.run()
