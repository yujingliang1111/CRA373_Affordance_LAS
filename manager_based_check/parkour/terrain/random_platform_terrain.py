from __future__ import annotations

import numpy as np
import trimesh


def random_platform_terrain(
    difficulty: float,
    cfg,
):
    
    meshes = []

    x = 0.0
    height = cfg.platform_height

    max_gap = (
        cfg.gap_width_range[0]
        + difficulty *
        (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    )


    for i in range(cfg.num_platforms):

        length = np.random.uniform(
            cfg.platform_length_range[0],
            cfg.platform_length_range[1],
        )

        y_offset = np.random.uniform(
            -0.5 * difficulty,
            0.5 * difficulty,
        )


        box = trimesh.creation.box(
            extents=(
                length,
                cfg.platform_width,
                cfg.platform_height,
            )
        )


        box.apply_translation(
            (
                x + length/2,
                y_offset,
                cfg.platform_height/2,
            )
        )


        meshes.append(box)


        gap = np.random.uniform(
            cfg.gap_width_range[0],
            max_gap,
        )

        x += length + gap



    mesh = trimesh.util.concatenate(meshes)


    # terrain origin
    origin = np.array([
        1.0,
        0.0,
        height,
    ])


    return mesh, origin