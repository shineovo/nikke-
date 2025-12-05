import tkinter as tk
import cv2
import numpy as np
import pyautogui
import threading
import time
import os
from PIL import Image

# 这些变量现在是全局常量，用于启动类
GAME_REGION = (44, 236, 471, 754)  # (左上角x, 左上角y, 宽, 高) - 您的精确值
ROWS = 16
COLS = 10
TARGET_SUM = 10
MATCH_THRESHOLD = 0.81  # 建议使用 0.70 避免匹配失败
TEMPLATE_DIR = "templates"  # 假设模板图片 1.png, 2.png 等就在当前脚本所在的子文件夹


# ================================================================

class GameOverlay:
    # 构造函数现在接受所有配置参数
    def __init__(self, game_region, rows, cols, target_sum, match_threshold, template_dir):

        # VVVVVV 修复 AttributeError 的关键：将配置变量保存为实例属性 VVVVVV
        self.GAME_REGION = game_region
        self.ROWS = rows
        self.COLS = cols
        self.TARGET_SUM = target_sum
        self.MATCH_THRESHOLD = match_threshold
        self.TEMPLATE_DIR = template_dir
        # ^^^^^^ 修复 AttributeError 的关键 ^^^^^^

        self.root = tk.Tk()
        self.root.title("Sum10 Assistant")

        # 1. 设置窗口属性
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", "white")
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.running = True
        self.solutions = []
        self.templates = self.load_templates()

    def is_overlapping(self, rect_a, rect_b):
        """检查两个矩形是否重叠。rect = (r1, c1, r2, c2)"""
        (ar1, ac1, ar2, ac2) = rect_a
        (br1, bc1, br2, bc2) = rect_b

        if ar1 > br2 or br1 > ar2:
            return False
        if ac1 > bc2 or bc1 > ac2:
            return False

        return True

    def load_templates(self):
        """从本地磁盘加载模板图片 (已使用 self.TEMPLATE_DIR)"""
        templates = {}
        # 使用 self.TEMPLATE_DIR
        print(f"尝试从目录 '{os.path.abspath(self.TEMPLATE_DIR)}' 加载模板...")

        for num in range(1, 10):
            file_name = os.path.join(self.TEMPLATE_DIR, f"{num}.png")

            if not os.path.exists(file_name):
                print(f"致命错误：未找到模板文件 {file_name}。请确保图片已正确命名并放置在指定目录下。")
                self.running = False
                return {}

            try:
                template_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                if template_img is None:
                    raise IOError(f"OpenCV 无法读取文件: {file_name}")

                templates[num] = template_img
            except Exception as e:
                print(f"致命错误：加载数字 {num} 的模板时发生读取错误。")
                print(f"错误信息: {e}")
                self.running = False
                return {}

        print("模板加载完成。")
        return templates

    def draw_box(self, x, y, w, h, color="red", thickness=3):
        self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=thickness)

    def clear_canvas(self):
        self.canvas.delete("all")

    def update_ui(self):
        """定时刷新UI，绘制提示框 (已修复像素精度)"""
        self.clear_canvas()
        if not self.running:
            self.root.destroy()
            return

        # VVVVVV 使用 self.GAME_REGION, self.ROWS, self.COLS VVVVVV
        cell_w = self.GAME_REGION[2] / self.COLS
        cell_h = self.GAME_REGION[3] / self.ROWS
        # ^^^^^^ 使用 self.GAME_REGION, self.ROWS, self.COLS ^^^^^^

        colors = ["#00FF00", "#00FFFF", "#FF00FF"]
        for i, ((r1, c1, r2, c2), score) in enumerate(self.solutions):
            # 最终的像素坐标使用 int() 转换，保证边界准确
            x = int(self.GAME_REGION[0] + c1 * cell_w)
            y = int(self.GAME_REGION[1] + r1 * cell_h)
            w = int((c2 - c1 + 1) * cell_w)
            h = int((r2 - r1 + 1) * cell_h)
            color = colors[i % len(colors)]
            self.draw_box(x, y, w, h, color=color, thickness=4 - i)

        self.root.after(50, self.update_ui)

    def start(self):
        if not self.running:
            print("因模板加载失败，程序停止启动。")
            return
        t = threading.Thread(target=self.processing_loop)
        t.daemon = True
        t.start()
        self.update_ui()
        self.root.mainloop()

    def processing_loop(self):
        """主处理循环 (最小面积优先 + 非重叠过滤)"""
        # VVVVVV 修复 AttributeError VVVVVV
        print(f"辅助已启动，监控区域: {self.GAME_REGION}")
        # ^^^^^^ 修复 AttributeError ^^^^^^
        print("按 Ctrl+C 在终端结束程序。")

        while self.running:
            try:
                start_time = time.time()
                matrix = self.get_grid_matrix()
                # print("--- 识别到的矩阵 ---") # 调试完成后应注释掉
                # print(matrix)
                # print("--------------------")
                found_solutions = self.find_all_solutions(matrix)

                # 策略更改：最小面积优先 (reverse=False)
                found_solutions.sort(key=lambda x: x[1], reverse=False)

                # 实施非重叠过滤 (只显示最多3个最小的、不重叠的解)
                filtered_solutions = []
                for solution, score in found_solutions:
                    overlaps = False
                    for existing_solution, _ in filtered_solutions:
                        if self.is_overlapping(solution, existing_solution):
                            overlaps = True
                            break
                    if not overlaps and len(filtered_solutions) < 3:
                        filtered_solutions.append((solution, score))

                self.solutions = filtered_solutions
                elapsed = time.time() - start_time
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                # print(f"处理循环中发生错误: {e}")
                time.sleep(1)

    def get_grid_matrix(self):
        """核心函数：使用模板匹配识别矩阵 (已使用 self.变量)"""
        # VVVVVV 使用 self.GAME_REGION VVVVVV
        screenshot = pyautogui.screenshot(region=self.GAME_REGION)
        # ^^^^^^ 使用 self.GAME_REGION ^^^^^^
        img_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

        # VVVVVV 使用 self.ROWS, self.COLS VVVVVV
        cell_h = self.GAME_REGION[3] // self.ROWS
        cell_w = self.GAME_REGION[2] // self.COLS
        matrix = np.zeros((self.ROWS, self.COLS), dtype=int)

        for r in range(self.ROWS):
            for c in range(self.COLS):
                # ^^^^^^ 使用 self.ROWS, self.COLS ^^^^^^
                x = c * cell_w
                y = r * cell_h

                # 向内缩进 5 像素
                roi = img_gray[y + 5:y + cell_h - 5, x + 5:x + cell_w - 5]

                best_match_score = -1
                best_match_num = 0

                for num, template in self.templates.items():
                    if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                        continue

                    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)

                    if max_val > best_match_score:
                        best_match_score = max_val
                        best_match_num = num

                # VVVVVV 使用 self.MATCH_THRESHOLD VVVVVV
                if best_match_score > self.MATCH_THRESHOLD:
                    # ^^^^^^ 使用 self.MATCH_THRESHOLD ^^^^^^
                    matrix[r, c] = best_match_num
                else:
                    matrix[r, c] = 0
        return matrix

    # ... (calculate_prefix_sum 和 get_submatrix_sum 略去，保持不变) ...

    def find_all_solutions(self, matrix):
        """查找所有和为 TARGET_SUM 的区域 (已使用 self.TARGET_SUM)"""
        rows, cols = matrix.shape
        solutions = []

        # 简化版：使用 np.sum 替代 Prefix Sum，对于小矩阵更快且不易出错
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        sub = matrix[r1:r2 + 1, c1:c2 + 1]
                        s = np.sum(sub)

                        # VVVVVV 使用 self.TARGET_SUM VVVVVV
                        if s == self.TARGET_SUM:
                            # ^^^^^^ 使用 self.TARGET_SUM ^^^^^^
                            score = (r2 - r1 + 1) * (c2 - c1 + 1)
                            solutions.append(((r1, c1, r2, c2), score))
                        # VVVVVV 使用 self.TARGET_SUM VVVVVV
                        elif s > self.TARGET_SUM:
                            # ^^^^^^ 使用 self.TARGET_SUM ^^^^^^
                            break
        return solutions


if __name__ == "__main__":
    # VVVVVV 启动时将全局变量传递给类 (修复 AttributeError 的关键) VVVVVV
    app = GameOverlay(
        game_region=GAME_REGION,
        rows=ROWS,
        cols=COLS,
        target_sum=TARGET_SUM,
        match_threshold=MATCH_THRESHOLD,
        template_dir=TEMPLATE_DIR
    )
    # ^^^^^^ 启动时将全局变量传递给类 ^^^^^^

    if app.running:
        app.start()