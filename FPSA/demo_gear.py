"""Interactive gear reshaping demo using the native slippage constraints."""

from __future__ import annotations

import numpy as np

try:
    from .FPSA import ShapeAugmentor
except ImportError:  # Direct execution: python FPSA/gear_validate.py
    from FPSA import ShapeAugmentor  # type: ignore


OBJ_PATH = "./data/objects/gear_extraction/gear/gear.obj"
GRASP_PATH = "./data/objects/gear_extraction/gear/grasp.yaml"

CONSTRAINT_IDS = [
    260, 257, 5437, 255, 469, 468, 467, 466, 464, 462, 459, 455, 453, 450,
    447, 275, 272, 269, 265, 263, 267, 1, 451, 452, 460, 461, 465, 258, 259,
    254, 270, 271, 273, 274, 276, 644, 648, 5463, 5405, 5454, 5416, 5320,
    5170, 5359, 5184, 5219, 5261, 5289, 5334,
]
RESHAPED_IDS = [5320, 5170, 5359, 5184, 5219, 5261, 5289, 5334]
DIRECTIONS = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.7071067811865476, 0.7071067811865476, 0.0],
        [0.7071067811865476, -0.7071067811865476, 0.0],
        [-0.7071067811865476, -0.7071067811865476, 0.0],
        [-0.7071067811865476, 0.7071067811865476, 0.0],
    ],
    dtype=np.float64,
)


def main() -> None:
    augmentor = ShapeAugmentor(OBJ_PATH, initial_grasp_path=GRASP_PATH)
    # Gear fixed points need a stronger boundary-condition weight. Other object
    # configs intentionally keep ShapeAugmentor's default value.
    augmentor.vertex_solve_params.bc_weight = 1e7

    vertices = augmentor.displacement_reshape(
        constraint_ids=CONSTRAINT_IDS,
        displace_idxs=RESHAPED_IDS,
        displacements=-0.03 * DIRECTIONS,
        max_iters=40,
        reshape_method="slippage",
        input_name="gear_slippage",
    )
    augmentor.write_augment_obj("reshaped_gear.obj", write_coacd=True)
    print("V_final:", vertices.shape)

    grasp, anchor, debug = augmentor.transfer_initial_grasp_guess(
        k_ring=3,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method="k_ring",
    )
    print("T_mesh_hand_tcp:\n", grasp["T_mesh_hand_tcp"])
    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=grasp["T_mesh_hand_tcp"],
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
        show_old_grasp=True,
    )


if __name__ == "__main__":
    main()
