import numpy as np
from .my_height_field_utils import my_height_field_to_mesh



# @my_height_field_to_mesh
# def my_stepping_stones_terrain(
#     difficulty,
#     cfg,
# ):

#     width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
#     length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

#     # difficulty:
#     # 0 -> shallow
#     # 1 -> deep

#     max_depth = 2.0   # meter

#     #test
#     difficulty = np.clip(difficulty, 0.08, 0.08)
#     depth = difficulty * max_depth

#     # convert meter to height field unit
#     hole_height = -depth / cfg.vertical_scale


#     hf = np.full(
#         (width_pixels, length_pixels),
#         hole_height,
#         dtype=np.float32,
#     )

#     # hf = np.full(
#     #     (width_pixels, length_pixels),
#     #     -400,
#     #     dtype=np.float32,
#     # )

#     hf[:20, :] = 0

#     hf[23:27,40:43] = 0.0
#     hf[29:33,40:43] = 0.0
#     hf[35:39,40:43] = 5.0
#     hf[40:51,40:48] = 5.0

#     hf[20:25,44:48] = 0.0
#     hf[25:29,44:48] = 0.0
#     hf[31:35,44:48] = 0.0
#     hf[37:41,44:48] = 5.0
    
    
#     hf[50:61,20:71] = 20.0#200.0

#     hf[60:,:] = 0

#     origin = np.array([
#         5 * cfg.horizontal_scale,
#         40 * cfg.horizontal_scale,
#         0.0,
#     ])
    
#     origin_pixel = (5, 44)

#     return hf.astype(np.int16), origin_pixel


# @my_height_field_to_mesh
# def my_stepping_stones_terrain(difficulty, cfg):


#     width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
#     length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

#     holes_depth = -2.0  # meter

#     d = np.clip(difficulty, 0.0, 1.0)

#     # =========================================================
#     # 1. 整個地形先全部設成 -2
#     # =========================================================

#     hf = np.full(
#         (width_pixels, length_pixels),
#         holes_depth / cfg.vertical_scale,
#         dtype=np.int16,
#     )

#     # =========================================================
#     # 2. 第一階段：只縮道路寬度
#     #
#     # difficulty 0.0 → 3.0 m
#     # difficulty 0.5 → 1.0 m
#     # =========================================================

#     width_d = min(d / 0.5, 1.0)

#     path_width = int(
#         30 - 20 * width_d
#     )

#     center_y = width_pixels // 2

#     y_start = center_y - path_width // 2
#     y_end = center_y + path_width // 2

#     # =========================================================
#     # 3. 第二階段：道路開始下降
#     #
#     # difficulty < 0.5 → 高度維持 0
#     # difficulty = 1.0 → 高度 -2
#     # =========================================================

#     height_d = max(
#         0.0,
#         (d - 0.5) / 0.5
#     )

#     path_height = -2.0 * height_d

#     # 建立道路
#     hf[:, y_start:y_end] = (
#         path_height / cfg.vertical_scale
#     )

#     # =========================================================
#     # 4. 起點平台
#     # =========================================================

#     hf[:20, :] = 0

#     # =========================================================
#     # 5. 你的石頭 —— 完全保留
#     # =========================================================

#     hf[23:27, 40:43] = 0.0
#     hf[29:33, 40:43] = 0.0
#     hf[35:39, 40:43] = 5.0
#     hf[40:51, 40:48] = 5.0

#     hf[20:25, 44:48] = 0.0
#     hf[25:29, 44:48] = 0.0
#     hf[31:35, 44:48] = 0.0
#     hf[37:41, 44:48] = 5.0

#     # =========================================================
#     # 6. 終點平台
#     # =========================================================

#     hf[50:61, 20:71] = 20.0

#     hf[60:, :] = 0

#     origin_pixel = (5, 44)

#     return hf.astype(np.int16), origin_pixel

#stairs
@my_height_field_to_mesh
def my_stepping_stones_terrain(difficulty, cfg):


    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    holes_depth = -2.0  # meter

    d = np.clip(difficulty, 0.0, 1.0)

    # =========================================================
    # 1. 整個地形先全部設成 -2
    # =========================================================

    hf = np.full(
        (width_pixels, length_pixels),
        holes_depth / cfg.vertical_scale,
        dtype=np.int16,
    )

    hf[20:61, 20:71] = difficulty * -5000.0
    # =========================================================

    hf[:20, :] = 0

    # =========================================================
    # 石頭
    # =========================================================

    hf[20:30, 30:50] = difficulty * 50.0
    hf[30:40, 30:50] = difficulty * 120.0
    hf[40:50, 30:50] = difficulty * 150.0
    hf[50:60, 30:50] = difficulty * 75.0


    # =========================================================
    # 6. 終點平台
    # =========================================================


    hf[60:, :] = 0

    

    origin_pixel = (5, 44)

    return hf.astype(np.int16), origin_pixel
