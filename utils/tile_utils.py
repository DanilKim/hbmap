import numpy as np


def image2tile(im, size: int) -> list:
    w = h = size
    tiles = [im[i:(i + h), j:(j + w), ...] for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    idxs = [(i, (i + h), j, (j + w)) for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    for idx, tile in enumerate(tiles):
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile = np.zeros_like(tiles[0])
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
            tiles[idx] = tile
    return tiles, idxs


def tile2image(tiles, idxs, size: list):
    seg = np.zeros([*size, tiles[0].shape[2]], dtype=tiles[0].dtype)
    for tile, (i1, i2, j1, j2) in zip(tiles, idxs):
        i2 = min(i2, size[0])
        j2 = min(j2, size[1])
        seg[i1:i2, j1:j2, :] = tile[:(i2 - i1), :(j2 - j1), :]
    return seg