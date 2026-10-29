import numpy as np
import trimesh


def gap_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generate a gap terrain.

    The terrain consists of two platforms separated by a gap.

    difficulty:
        0 ~ 1 controls gap width.
    """

    # interpolate gap size by difficulty
    gap_width = (
        cfg.gap_width_range[0]
        + difficulty
        * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    )

    platform_width = cfg.platform_width
    platform_length = cfg.platform_length
    height = cfg.platform_height


    meshes_list = list()


    # -------------------------
    # Left platform (takeoff)
    # -------------------------

    left_dims = (
        platform_length,
        platform_width,
        height,
    )

    left_pos = (
        -gap_width / 2 - platform_length / 2,
        0,
        height / 2,
    )

    left_platform = trimesh.creation.box(
        left_dims,
        trimesh.transformations.translation_matrix(left_pos),
    )


    # -------------------------
    # Right platform (landing)
    # -------------------------

    right_dims = (
        platform_length,
        platform_width,
        height,
    )

    right_pos = (
        gap_width / 2 + platform_length / 2,
        0,
        height / 2,
    )

    right_platform = trimesh.creation.box(
        right_dims,
        trimesh.transformations.translation_matrix(right_pos),
    )


    meshes_list += [
        left_platform,
        right_platform,
    ]


    # terrain origin
    origin = np.array([
        -1.5,
        0.0,
        height,
    ])



    return meshes_list, origin