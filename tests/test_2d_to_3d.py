#!/usr/bin/env python3
"""
阶段 4：2D → 3D 转换验证
- 验证像素坐标 + 深度 → 3D 世界坐标的转换正确性
- 验证 3D → 2D 反投影误差
"""

import numpy as np
import cv2
import os

OUTPUT_DIR = "/home/xshan/tracking/tests/outputs"


def pixel_to_3d(u, v, depth, K):
    """
    将像素坐标转换为相机坐标系下的 3D 坐标

    Args:
        u, v: 像素坐标
        depth: 深度图 (H, W) 单位：米
        K: 3x3 内参矩阵

    Returns:
        (x, y, z) 相机坐标系下的 3D 坐标，单位：米
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 获取深度值
    v_int, u_int = int(round(v)), int(round(u))
    z = depth[v_int, u_int]

    if z <= 0:
        return None

    # 反投影
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z])


def project_3d_to_2d(point_3d, K):
    """
    将 3D 坐标投影到像素坐标

    Args:
        point_3d: (x, y, z) 3D 坐标
        K: 3x3 内参矩阵

    Returns:
        (u, v) 像素坐标
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = point_3d

    if z <= 0:
        return None

    u = fx * x / z + cx
    v = fy * y / z + cy

    return np.array([u, v])


def main():
    print("=" * 50)
    print("阶段 4：2D → 3D 转换验证")
    print("=" * 50)

    # 加载数据
    print("\n[1] 加载数据...")
    rgb = cv2.imread(os.path.join(OUTPUT_DIR, "frame_000_rgb.png"))
    depth = np.load(os.path.join(OUTPUT_DIR, "frame_000_depth_raw.npy"))

    K = np.array([
        [605.903015, 0.0, 323.908539],
        [0.0, 605.985046, 248.735535],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    print(f"    RGB: {rgb.shape}")
    print(f"    Depth: {depth.shape}")
    print(f"    K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # 测试点（阶段 3 使用的点 + 图像中心）
    test_points = [
        (320, 240, "图像中心"),
        (320, 350, "纸盒中心"),
        (280, 380, "纸盒左边"),
        (360, 380, "纸盒右边"),
        (500, 300, "支架位置"),
    ]

    print(f"\n[2] 测试 {len(test_points)} 个点的 2D→3D→2D 转换...")

    results = []
    log_lines = []
    log_lines.append("2D → 3D 转换验证结果\n")
    log_lines.append("=" * 40 + "\n\n")

    all_pass = True

    for u, v, name in test_points:
        print(f"\n    点: {name} ({u}, {v})")
        log_lines.append(f"点: {name} ({u}, {v})\n")

        # 2D → 3D
        point_3d = pixel_to_3d(u, v, depth, K)

        if point_3d is None:
            print(f"      ❌ 深度无效")
            log_lines.append(f"  深度无效\n\n")
            results.append((u, v, name, None, None, None))
            continue

        x, y, z = point_3d
        print(f"      3D 坐标: x={x:.4f}m, y={y:.4f}m, z={z:.4f}m")
        log_lines.append(f"  3D: x={x:.4f}m, y={y:.4f}m, z={z:.4f}m\n")

        # 3D → 2D
        point_2d = project_3d_to_2d(point_3d, K)
        u2, v2 = point_2d

        # 计算误差
        error = np.sqrt((u2 - u) ** 2 + (v2 - v) ** 2)
        pass_test = error < 1.0

        if not pass_test:
            all_pass = False

        status = "✅" if pass_test else "❌"
        print(f"      反投影: ({u2:.2f}, {v2:.2f})")
        print(f"      误差: {error:.4f} px {status}")

        log_lines.append(f"  反投影: ({u2:.2f}, {v2:.2f})\n")
        log_lines.append(f"  误差: {error:.4f} px {status}\n\n")

        results.append((u, v, name, point_3d, point_2d, error))

    # 保存日志
    with open(os.path.join(OUTPUT_DIR, "3d_coordinates.txt"), 'w') as f:
        f.writelines(log_lines)
    print(f"\n    已保存: 3d_coordinates.txt")

    # 创建可视化
    print(f"\n[3] 创建可视化...")
    vis = rgb.copy()

    for u, v, name, point_3d, point_2d, error in results:
        if point_3d is None:
            # 无效点 - 黄色
            cv2.circle(vis, (u, v), 8, (0, 255, 255), -1)
            cv2.putText(vis, "invalid", (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # 原始点 - 红色
            cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)

            # 反投影点 - 蓝色
            u2, v2 = int(round(point_2d[0])), int(round(point_2d[1]))
            cv2.circle(vis, (u2, v2), 8, (255, 0, 0), 2)

            # 标注 3D 坐标
            z = point_3d[2]
            cv2.putText(vis, f"z={z:.2f}m", (u + 10, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis, f"e={error:.2f}px", (u + 10, v + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 添加图例
    cv2.putText(vis, "Red: Original 2D", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, "Blue: Reprojected 2D", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "3d_projection_test.png"), vis)
    print(f"    已保存: 3d_projection_test.png")

    # 汇总
    print(f"\n[4] 汇总...")
    valid_results = [r for r in results if r[3] is not None]
    if valid_results:
        errors = [r[5] for r in valid_results]
        max_error = max(errors)
        mean_error = np.mean(errors)
        print(f"    有效点数: {len(valid_results)}/{len(results)}")
        print(f"    最大误差: {max_error:.4f} px")
        print(f"    平均误差: {mean_error:.4f} px")

        # 3D 坐标范围
        z_values = [r[3][2] for r in valid_results]
        print(f"    深度范围: {min(z_values):.3f}m - {max(z_values):.3f}m")

    print("\n" + "=" * 50)
    if all_pass:
        print("阶段 4 完成! ✅ 所有反投影误差 < 1 像素")
    else:
        print("阶段 4 完成! ⚠️ 部分点反投影误差较大")
    print("=" * 50)

    return all_pass


if __name__ == "__main__":
    main()
