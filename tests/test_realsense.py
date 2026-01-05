#!/usr/bin/env python3
"""
阶段 1：RealSense 数据采集验证
- 连接 RealSense 相机
- 读取 RGB 和深度帧
- 保存测试图像和内参
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os

OUTPUT_DIR = "/home/xshan/tracking/tests/outputs"

def main():
    print("=" * 50)
    print("阶段 1：RealSense 数据采集验证")
    print("=" * 50)

    # 配置 RealSense 流
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用 RGB 和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动流
    print("\n[1] 启动 RealSense 流...")
    profile = pipeline.start(config)

    # 获取深度传感器的深度比例
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"    深度比例: {depth_scale} (米/单位)")

    # 获取内参
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    print(f"\n[2] 相机内参:")
    print(f"    分辨率: {intrinsics.width} x {intrinsics.height}")
    print(f"    fx: {intrinsics.fx:.2f}")
    print(f"    fy: {intrinsics.fy:.2f}")
    print(f"    cx: {intrinsics.ppx:.2f}")
    print(f"    cy: {intrinsics.ppy:.2f}")

    # 构建内参矩阵
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    # 创建对齐对象（将深度对齐到彩色）
    align = rs.align(rs.stream.color)

    # 跳过前 30 帧让相机稳定
    print("\n[3] 等待相机稳定...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # 采集帧
    print("\n[4] 采集帧数据...")
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("错误: 无法获取帧数据!")
        pipeline.stop()
        return False

    # 转换为 numpy 数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 深度转换为米
    depth_meters = depth_image.astype(np.float32) * depth_scale

    print(f"    RGB 图像形状: {color_image.shape}")
    print(f"    深度图像形状: {depth_image.shape}")
    print(f"    深度范围: {depth_meters[depth_meters > 0].min():.3f}m - {depth_meters.max():.3f}m")

    # 保存文件
    print(f"\n[5] 保存文件到 {OUTPUT_DIR}/")

    # RGB 图像
    rgb_path = os.path.join(OUTPUT_DIR, "frame_000_rgb.png")
    cv2.imwrite(rgb_path, color_image)
    print(f"    - frame_000_rgb.png")

    # 深度可视化（伪彩色）
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    depth_vis_path = os.path.join(OUTPUT_DIR, "frame_000_depth.png")
    cv2.imwrite(depth_vis_path, depth_colormap)
    print(f"    - frame_000_depth.png")

    # 原始深度数据
    depth_raw_path = os.path.join(OUTPUT_DIR, "frame_000_depth_raw.npy")
    np.save(depth_raw_path, depth_meters)
    print(f"    - frame_000_depth_raw.npy")

    # 内参矩阵
    intrinsic_path = os.path.join(OUTPUT_DIR, "intrinsic.txt")
    with open(intrinsic_path, 'w') as f:
        f.write("# RealSense D435IF 内参矩阵 K (3x3)\n")
        f.write(f"# 分辨率: {intrinsics.width} x {intrinsics.height}\n")
        f.write(f"# 深度比例: {depth_scale}\n\n")
        f.write("K = [\n")
        for row in K:
            f.write(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],\n")
        f.write("]\n\n")
        f.write(f"fx = {intrinsics.fx:.6f}\n")
        f.write(f"fy = {intrinsics.fy:.6f}\n")
        f.write(f"cx = {intrinsics.ppx:.6f}\n")
        f.write(f"cy = {intrinsics.ppy:.6f}\n")
    print(f"    - intrinsic.txt")

    # 停止流
    pipeline.stop()

    print("\n" + "=" * 50)
    print("阶段 1 完成!")
    print("=" * 50)

    return True

if __name__ == "__main__":
    main()
