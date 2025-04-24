grid_positions = {
    "a": (0, 0),
    "b": (0, 1),
    "c": (0, 2),
    "d": (1, 0),
    "e": (1, 1),
    "f": (1, 2),
    "g": (2, 0),
    "h": (2, 1),
    "i": (2, 2),
}
# 从坐标反向映射到位置字母
position_to_letter = {v: k for k, v in grid_positions.items()}


def _precompute_positions(layout, length):
    """预计算特定长度和布局的所有可能位置"""
    positions = []
    if layout == "horizontal":
        for x in range(3):
            for y in range(4 - length):
                pos = [(x, y + i) for i in range(length)]
                positions.append([position_to_letter[p] for p in pos])
    else:  # vertical
        for x in range(4 - length):
            for y in range(3):
                pos = [(x + i, y) for i in range(length)]
                positions.append([position_to_letter[p] for p in pos])
    return positions


position_combinations = {}
for block_length in [1, 2, 3]:
    position_combinations[block_length] = {
        "horizontal": _precompute_positions("horizontal", block_length),
        "vertical": _precompute_positions("vertical", block_length),
    }

print(position_combinations)