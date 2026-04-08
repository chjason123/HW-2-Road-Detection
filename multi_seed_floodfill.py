import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# 全局變數
original_image, resized_image, temp_image = None, None, None
seeds = []


# 縮放影像
def resize_image(image, min_size=1024):
    h, w = image.shape[:2]
    if max(h, w) < min_size:
        scale = min_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"Resizing image from ({h}, {w}) to ({new_h}, {new_w})")
        return cv2.resize(image, (new_w, new_h))
    return image


# 滑鼠事件回調：新增種子點
def mouse_handler(event, x, y, flags, param):
    global seeds, temp_image
    if event == cv2.EVENT_LBUTTONDOWN:  # 滑鼠左鍵點擊作為種子點
        seeds.append((x, y))
        cv2.circle(temp_image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # 標記種子點
        cv2.imshow("Image", temp_image)
        print(f"Seed added at {x}, {y}")


# 執行 OpenCV 的 Flood Fill (支持逐步顯示)
def perform_flood_fill():
    global temp_image, seeds

    # 獲取滑桿參數
    use_resize = cv2.getTrackbarPos("Resize Image (Checkbox)", "UI") == 1
    visualize = cv2.getTrackbarPos("Visualize Process (Checkbox)", "UI") == 1
    delay = 101 - cv2.getTrackbarPos("Flood Fill Speed", "UI")
    lo_diff = cv2.getTrackbarPos("loDiff", "UI")
    up_diff = cv2.getTrackbarPos("upDiff", "UI")

    print(f"Flood Fill starting with Resize={use_resize}, Visualize={visualize}, "
          f"Delay={delay}, loDiff={lo_diff}, upDiff={up_diff}")

    # 選擇是否使用縮放影像
    target_image = resized_image if use_resize else original_image
    flood_filled_image = target_image.copy()

    for seed in seeds:
        print(f"Flood Fill at seed: {seed}")
        # 準備遮罩
        mask = np.zeros((flood_filled_image.shape[0] + 2, flood_filled_image.shape[1] + 2), dtype=np.uint8)

        if visualize:
            # 逐步顯示填充過程
            for step in range(10):  # 模擬逐步執行（分 10 次填充）
                _, _, _, rect = cv2.floodFill(flood_filled_image, mask, seed,
                                              (0, 255, 0), (lo_diff,) * 3, (up_diff,) * 3, flags=8)
                print(f"Partial Flood Fill step={step}, filled area={rect}")
                cv2.imshow("Flood Fill Process", flood_filled_image)
                cv2.waitKey(delay)

        else:
            # 一次性完成 Flood Fill
            _, _, _, rect = cv2.floodFill(flood_filled_image, mask, seed,
                                          (0, 255, 0), (lo_diff,) * 3, (up_diff,) * 3, flags=8)
            print(f"Flood Fill completed, filled rect={rect}")

    # 最終結果顯示
    cv2.imshow("Flood Filled Image", flood_filled_image)


# 重置影像
def reset_image():
    global temp_image, seeds
    seeds.clear()
    temp_image = original_image.copy()  # 還原影像
    cv2.imshow("Image", temp_image)
    print("Image reset to original.")


# 主控制循環
if __name__ == "__main__":
    # 載入影像
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="選擇影像檔案")
    if not file_path:
        print("未選擇影像，程式結束")
        exit()
    original_image = cv2.imread(file_path)

    # 生成縮放影像
    resized_image = resize_image(original_image)
    temp_image = original_image.copy()
    cv2.imshow("Image", temp_image)

    # 鼠標與 UI 控制
    cv2.setMouseCallback("Image", mouse_handler)
    cv2.namedWindow("UI")
    cv2.createTrackbar("Resize Image (Checkbox)", "UI", 0, 1, lambda x: None)
    cv2.createTrackbar("Visualize Process (Checkbox)", "UI", 0, 1, lambda x: None)
    cv2.createTrackbar("Flood Fill Speed", "UI", 50, 100, lambda x: None)
    cv2.createTrackbar("loDiff", "UI", 10, 255, lambda x: None)
    cv2.createTrackbar("upDiff", "UI", 10, 255, lambda x: None)

    print("操作指南：")
    print("- 種子點選擇（鼠標左鍵點擊影像）")
    print("- 使用滑桿調整參數：")
    print("    - Resize Image：影像縮放開關")
    print("    - Visualize Process：逐步填充可視化")
    print("    - loDiff 與 upDiff：容容範圍調整")
    print("- 按 'f' 執行 Flood Fill，按 'r' 重置影像")
    print("- 按 'ESC' 結束程式")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            if not seeds:
                print("請先選擇種子點再執行 Flood Fill")
            else:
                perform_flood_fill()
        elif key == ord('r'):
            reset_image()
        elif key == 27:  # ESC 鍵 - 結束程式
            print("程序結束")
            break

    cv2.destroyAllWindows()