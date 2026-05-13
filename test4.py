import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    height, width = img.shape
    n_blocks_height = int(np.ceil(height / block_size))
    n_blocks_width = int(np.ceil(width / block_size))
    blocks = np.zeros((n_blocks_height, n_blocks_width, block_size, block_size), dtype=np.uint8)
    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[i, j, :, :] = block
    return blocks

def manual_lbp(block):
    lbp = np.zeros_like(block, dtype=np.uint8)
    for i in range(1, block.shape[0] - 1):
        for j in range(1, block.shape[1] - 1):
            center = block[i, j]
            binary_string = "".join(['1' if block[i + dx, j + dy] >= center else '0'
                                     for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                                    (1, 1), (1, 0), (1, -1), (0, -1)]])
            lbp[i, j] = int(binary_string, 2)
    return lbp

def blocks_to_hist(blocks):
    num_blocks_y, num_blocks_x, _, _ = blocks.shape
    hist = np.zeros((num_blocks_y, num_blocks_x, 256), dtype=np.float32)
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            lbp_block = manual_lbp(blocks[y, x])
            hist[y, x] = np.histogram(lbp_block.ravel(), bins=256, range=(0, 256))[0]
            hist[y, x] = hist[y, x] / hist[y, x].sum()  # Normalize
    return hist

def bfs_with_feature(_x, _y, _hist, target_feature, similarity=0.85):
    num_blocks_y, num_blocks_x, _ = _hist.shape
    queue = [(_x, _y)]
    _result = []
    visited = set()
    visited.add((_x, _y))
    while queue:
        currentVertex = queue.pop(0)
        for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_x, new_y = currentVertex[0] + di, currentVertex[1] + dj
            if new_x < 0 or new_x >= num_blocks_x or new_y < 0 or new_y >= num_blocks_y:
                continue
            if (new_x, new_y) in visited:
                continue
            # 只比對區塊內是否包含最常見的特徵值
            if _hist[new_y, new_x, target_feature] >= similarity * _hist[new_y, new_x].sum():
                _result.append((new_x, new_y))
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))
    return _result

def display(img, cmap=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('road_2.png')
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel 邊緣檢測
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)

    block_size = 16
    blocks = to_blocks(sobel, block_size)
    hist = blocks_to_hist(blocks)
    num_blocks_y, num_blocks_x, _ = hist.shape

    # 計算底部三排區塊的 LBP 直方圖，並找出最頻繁的特徵值
    bottom_blocks = hist[-3:, :, :]  # 選擇底部三排
    flattened_hist = bottom_blocks.reshape(-1, 256)
    most_common_feature = np.argmax(flattened_hist.sum(axis=0))  # 找出最常見的特徵值

    # 使用底部中間區塊作為 BFS 起始點
    start_x, start_y = num_blocks_x // 2, num_blocks_y - 1
    result = bfs_with_feature(start_x, start_y, hist, most_common_feature, similarity=0.273)

    # 將結果還原成影像
    img2 = np.zeros((num_blocks_y * block_size, num_blocks_x * block_size, 3), dtype=np.uint8)
    for j, i in result:
        img2[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size, 0] = 255
    result_img = cv2.add(img, img2)
    cv2.imshow('LBP Road Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
