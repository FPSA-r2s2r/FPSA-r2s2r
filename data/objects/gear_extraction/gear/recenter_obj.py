#!/usr/bin/env python3
import argparse
import numpy as np
import trimesh


def recenter_obj(input_path: str, output_path: str) -> None:
    mesh = trimesh.load(input_path, force="mesh", process=False)

    if mesh.vertices.size == 0:
        raise ValueError("OBJ 文件中没有顶点。")

    # Use the mean position of all vertices as the new origin.
    origin = np.mean(mesh.vertices, axis=0)

    # Translate all vertices so the origin moves to (0, 0, 0).
    mesh.apply_translation(-origin)

    mesh.export(output_path)

    print(f"原始顶点中心: {origin}")
    print(f"已保存到: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 OBJ 模型的顶点平均位置移动到坐标原点。"
    )
    parser.add_argument("input", help="输入 .obj 文件路径")
    parser.add_argument("output", help="输出 .obj 文件路径")
    args = parser.parse_args()

    recenter_obj(args.input, args.output)


if __name__ == "__main__":
    main()
