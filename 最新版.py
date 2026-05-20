import cv2
import numpy as np

# 用於滑鼠繪圖的全局變數
drawing = False
mask_img = None

def mouse_callback(event, x, y, flags, param):
    """ 滑鼠回呼函式：記錄滑鼠劃過的區域 """
    global drawing, mask_img
    brush_size = 10 # 畫筆粗細
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask_img, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask_img, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def to_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    height, width = img.shape
    n_blocks_height = height // block_size
    n_blocks_width = width // block_size
    blocks = np.zeros((n_blocks_height, n_blocks_width, block_size, block_size), dtype=np.uint8)
    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            blocks[i, j, :, :] = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
    return blocks

def vectorized_lbp(block):
    h, w = block.shape
    lbp = np.zeros_like(block, dtype=np.uint8)
    if h < 3 or w < 3:
        return lbp
    center = block[1:-1, 1:-1]
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for power, (dx, dy) in enumerate(neighbors):
        neighbor_pixels = block[1+dx:h-1+dx, 1+dy:w-1+dy]
        lbp[1:-1, 1:-1] |= ((neighbor_pixels >= center).astype(np.uint8) << (7 - power))
    return lbp

def blocks_to_hist(blocks):
    num_blocks_y, num_blocks_x, _, _ = blocks.shape
    hist = np.zeros((num_blocks_y, num_blocks_x, 256), dtype=np.float32)
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            lbp_block = vectorized_lbp(blocks[y, x])
            valid_pixels = lbp_block[1:-1, 1:-1].ravel()
            if len(valid_pixels) > 0:
                hist_counts = np.histogram(valid_pixels, bins=256, range=(0, 256))[0]
                sum_counts = hist_counts.sum()
                if sum_counts > 0:
                    hist[y, x] = hist_counts / sum_counts
    return hist

def bfs_with_features(start_x, start_y, _hist, target_features, similarity=0.25):
    num_blocks_y, num_blocks_x, _ = _hist.shape
    queue = [(start_x, start_y)]
    _result = [(start_x, start_y)]
    visited = set([(start_x, start_y)])
    
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_x, new_y = cx + dx, cy + dy
            if new_x < 0 or new_x >= num_blocks_x or new_y < 0 or new_y >= num_blocks_y:
                continue
            if (new_x, new_y) in visited:
                continue
            visited.add((new_x, new_y))
            
            total_share = 0.0
            for f in target_features:
                total_share += _hist[new_y, new_x, f]
            
            if total_share >= similarity:
                _result.append((new_x, new_y))
                queue.append((new_x, new_y))
    return _result

if __name__ == '__main__':
    # 模式選擇
    print("="*40)
    print("  LBP + BFS 道路特徵偵測系統 (自動儲存結果版)")
    print("="*40)
    print("請選擇特徵分析模式：")
    print("[1] 自動偵測模式 (自動抓取影像底部三排特徵)")
    print("[2] 手動標記模式 (用滑鼠在畫面上塗抹特徵候選區)")
    mode = input("請輸入模式編號 (1 或 2): ").strip()
    
    if mode not in ['1', '2']:
        print("輸入錯誤，預設切換至 [1] 自動偵測模式。")
        mode = '1'

    img = cv2.imread('road_2.png')
    if img is None:
        print("錯誤：找不到影像檔案 'road_1.jpg'")
        exit()
        
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    h_img, w_img, _ = img.shape
    
    if mode == '2':
        mask_img = np.zeros((h_img, w_img), dtype=np.uint8)
        win_name = "[Step 1] Paint Blocks (Press SPACE to confirm)"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback)
        
        print("\n[手動模式啟動] 請用滑鼠塗抹道路區域，完成後請按『空白鍵』。")
        while True:
            display_img = img.copy()
            display_img[mask_img > 0] = [0, 255, 0]
            cv2.imshow(win_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                break
        cv2.destroyWindow(win_name)

    # 影像共通處理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)

    # 切分區塊 (目前設定 block_size = 16)
    block_size = 16

    blocks = to_blocks(sobel, block_size)
    hist = blocks_to_hist(blocks)
    num_blocks_y, num_blocks_x, _ = hist.shape

    total_target_hist = np.zeros(256, dtype=np.float32)
    start_x, start_y = None, None
    source_blocks = set()

    # 分支邏輯：收集直方圖特徵
    feature_thresholds = {} 

    if mode == '1':
        print("\n[自動模式] 正在分析影像底部紋理...")
        bottom_blocks_hist = hist[-3:, :, :]
        total_target_hist = bottom_blocks_hist.reshape(-1, 256).sum(axis=0)
        start_x = num_blocks_x // 2
        start_y = num_blocks_y - 1
        
        for y in range(num_blocks_y - 3, num_blocks_y):
            for x in range(num_blocks_x):
                source_blocks.add((x, y))
        
        # 自動模式維持原邏輯簡化處理：取全局第一名，給予預設相似度
        total_target_hist[0] = 0
        max_f = int(np.argsort(total_target_hist)[-1])
        feature_thresholds[max_f] = 0.11 
        
    elif mode == '2':
        mask_blocks = to_blocks(mask_img, block_size)
        painted_block_count = 0
        
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                if np.any(mask_blocks[y, x] > 0):
                    start_x, start_y = x, y
                    painted_block_count += 1
                    source_blocks.add((x, y))
                    
                    block_hist = hist[y, x].copy()
                    block_hist[0] = 0    
                    block_hist[255] = 0  
                    
                    max_feature = int(np.argmax(block_hist))
                    max_share = float(block_hist[max_feature])
                    
                    if max_share >= 0.03:
                        if max_feature not in feature_thresholds:
                            feature_thresholds[max_feature] = max_share
                        else:
                            if max_share < feature_thresholds[max_feature]:
                                feature_thresholds[max_feature] = max_share

        if painted_block_count == 0 or start_x is None:
            print("警告：您沒有塗抹到任何有效的區塊區域！程式強制結束。")
            exit()
            
        print(f"\n[手動模式] 劃到的區塊共計 {painted_block_count} 個。")

    print("\n📋 最終篩選出的道路特徵白名單與【最低】檢驗門檻：")
    for f, th in feature_thresholds.items():
        print(f"  - 特徵碼 [{f}]: 必須大於等於 {th*100:.2f}%")

    # -------------------------------------------------------------
    # 執行 BFS 區域增長
    # -------------------------------------------------------------
    print("\n🚀 開始依據新門檻進行道路辨識...")
    
    queue = [(start_x, start_y)]
    result = [(start_x, start_y)]
    visited = {(start_x, start_y)}
    
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_x, new_y = cx + dx, cy + dy
            if new_x < 0 or new_x >= num_blocks_x or new_y < 0 or new_y >= num_blocks_y:
                continue
            if (new_x, new_y) in visited:
                continue
            visited.add((new_x, new_y))
            
            is_road_block = False
            for target_f, min_required_share in feature_thresholds.items():
                neighbor_share = hist[new_y, new_x, target_f]
                if neighbor_share >= min_required_share:
                    is_road_block = True
                    break
            
            if is_road_block:
                result.append((new_x, new_y))
                queue.append((new_x, new_y))

    # -------------------------------------------------------------
    # 二次加工：網格空洞填補邏輯
    # -------------------------------------------------------------
    print("\n盤點網格空洞，正在填補被馬路包圍的空格...")
    
    grid_mask = np.zeros((num_blocks_y, num_blocks_x), dtype=np.uint8)
    for bx, by in result:
        grid_mask[by, bx] = 255

    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_grid_mask = grid_mask.copy()
    cv2.drawContours(filled_grid_mask, contours, -1, 255, thickness=cv2.FILLED)

    filled_holes_set = set() 
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            if filled_grid_mask[y, x] == 255 and grid_mask[y, x] == 0:
                filled_holes_set.add((x, y))
                result.append((x, y)) 

    print(f"整理完成！共發現並填補了 {len(filled_holes_set)} 個被包圍的網格空格。")
    result_set = set(result)

    # -------------------------------------------------------------
    # 建立基礎的藍色(原道路) 與 紅色(填補區) 半透明道路遮罩
    # -------------------------------------------------------------
    img2 = np.zeros((num_blocks_y * block_size, num_blocks_x * block_size, 3), dtype=np.uint8)
    for bx, by in result:
        y_start, y_end = by * block_size, (by + 1) * block_size
        x_start, x_end = bx * block_size, (bx + 1) * block_size
        
        if (bx, by) in filled_holes_set:
            img2[y_start:y_end, x_start:x_end, 2] = 255 # 被包圍的空格填紅色
        else:
            img2[y_start:y_end, x_start:x_end, 0] = 255 # 原始馬路填藍色

    img_cropped = img[0:num_blocks_y * block_size, 0:num_blocks_x * block_size].copy()
    
    # 圖一：呈現乾淨的 BFS 道路偵測 + 紅色填補結果 (無文字特徵標記)
    clean_result_img = cv2.add(img_cropped, img2)

    # 🎯 儲存沒有特徵值標記的乾淨圖片
    save_filename = 'road_2_out.png'
    success = cv2.imwrite(save_filename, clean_result_img)
    if success:
        print(f"\n💾 成功將無特徵值標記的乾淨結果圖儲存至: '{save_filename}'")
    else:
        print(f"\n❌ 錯誤：圖片儲存失敗。")

    # 圖二：包含特徵值、網格與占比的細節加工圖
    labeled_result_img = clean_result_img.copy()

    # 針對字體大小與定位做微調
    font_scale = 0.22
    text_y_offset1 = 7

    print("\n📊 正在繪製特徵標記細節圖...")
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            y_start = y * block_size
            x_start = x * block_size
            
            block_hist = hist[y, x].copy()
            block_hist[0] = 0  
            orig_max_feature = np.argmax(block_hist)
            orig_max_share = block_hist[orig_max_feature]

            # -----------------------------------------------------------------
            # 情況 A：特徵來源區塊（綠框）
            # -----------------------------------------------------------------
            if (x, y) in source_blocks:
                cv2.rectangle(labeled_result_img, (x_start, y_start), (x_start + block_size, y_start + block_size), (0, 255, 0), 1)
                
                src_valid_features = []
                for f, th in feature_thresholds.items():
                    if block_hist[f] > 0:
                        src_valid_features.append((f, block_hist[f]))
                
                if src_valid_features:
                    best_src_f, best_src_s = max(src_valid_features, key=lambda x: x[1])
                    cv2.putText(labeled_result_img, f"{best_src_f}", (x_start + 1, y_start + text_y_offset1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
            
            # -----------------------------------------------------------------
            # 情況 B：道路區塊 (原本 BFS 延伸的藍框 或 後來填補的紅框)
            # -----------------------------------------------------------------
            elif (x, y) in result_set:
                if (x, y) in filled_holes_set:
                    cv2.rectangle(labeled_result_img, (x_start, y_start), (x_start + block_size, y_start + block_size), (0, 0, 255), 1)
                    cv2.putText(labeled_result_img, "Hole", (x_start + 1, y_start + text_y_offset1), cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.02, (0, 0, 255), 1)
                else:
                    cv2.rectangle(labeled_result_img, (x_start, y_start), (x_start + block_size, y_start + block_size), (255, 100, 0), 1)
                    
                    passed_features = []
                    for target_f, min_required_share in feature_thresholds.items():
                        neighbor_share = block_hist[target_f]
                        if neighbor_share >= min_required_share:
                            passed_features.append((target_f, neighbor_share, min_required_share))
                    
                    if passed_features:
                        best_f, best_s, min_s = max(passed_features, key=lambda x: x[1])
                        cv2.putText(labeled_result_img, f"{best_f}", (x_start + 1, y_start + text_y_offset1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    else:
                        cv2.putText(labeled_result_img, "?", (x_start + 1, y_start + text_y_offset1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            
            # -----------------------------------------------------------------
            # 情況 C：非道路區塊 
            # -----------------------------------------------------------------
            else:
                if orig_max_feature == 255 and orig_max_share > 0.2:
                     cv2.putText(labeled_result_img, "255", (x_start + 1, y_start + text_y_offset1), cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.05, (100, 100, 100), 1)

    print("標記完成。")
    
    # 同時開啟兩張圖進行比對
    mode_title = "Auto Bottom" if mode == '1' else "Manual Painting"
    cv2.imshow(f'[1. Clean Result] Mode: {mode_title}', clean_result_img)
    cv2.imshow(f'[2. Labeled Detail] Mode: {mode_title}', labeled_result_img)
    
    print("\n系統已成功開啟兩個視窗並完成儲存！")
    print("視窗 1 為乾淨偵測結果(含紅色填補)，視窗 2 為特徵細節標記。請按任意鍵結束程式。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()