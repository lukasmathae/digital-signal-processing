import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import os
# from ai_model import analyze_image  # Your AI model logic

output_csv = "output.csv"
captured_image_path = "captured_image.jpg"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Analyzer")

        # Canvas for image/video
        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.pack()

        # Main buttons
        self.capture_btn = tk.Button(root, text="Take Picture", command=self.capture_image)
        self.retake_btn = tk.Button(root, text="Retake Picture", command=self.retake_picture)

        self.capture_btn.pack(pady=5)
        self.retake_btn.pack_forget()

        # Image editing buttons
        self.buttons_frame = tk.Frame(root)
        self.rotate_btn = tk.Button(self.buttons_frame, text="Rotate Image", command=self.rotate_image)
        self.crop_btn = tk.Button(self.buttons_frame, text="Crop Image", command=self.enable_crop_mode)
        self.apply_crop_btn = tk.Button(self.buttons_frame, text="Apply Crop", command=self.apply_crop)
        self.undo_crop_btn = tk.Button(self.buttons_frame, text="Undo Crop", command=self.undo_crop)
        self.analyze_btn = tk.Button(self.buttons_frame, text="Analyze Image", command=self.analyze)
        self.save_btn = tk.Button(self.buttons_frame, text="Save to CSV", command=self.save_to_csv)

        for btn in [self.rotate_btn, self.crop_btn, self.apply_crop_btn,
                    self.undo_crop_btn, self.analyze_btn, self.save_btn]:
            btn.pack(pady=2)

        self.original_image = None
        self.image = None
        self.result = None
        self.tk_img = None

        # Crop logic
        self.crop_box = None
        self.handles = {}
        self.crop_mode = False
        self.dragging = None
        self.drag_offset = (0, 0)
        self.crop_box_coords = []

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.video_loop_id = None
        self.previewing = True  # controls preview state
        self.update_video_feed()

    def update_video_feed(self):
        if self.previewing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self.tk_img = ImageTk.PhotoImage(img.resize((400, 300)))
                self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
        self.video_loop_id = self.root.after(10, self.update_video_feed)

    def capture_image(self):
        if self.current_frame is not None:
            self.previewing = False  # stop updating video
            cv2.imwrite(captured_image_path, self.current_frame)
            self.image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
            self.original_image = self.image.copy()
            self.show_static_image()

            self.capture_btn.pack_forget()
            self.retake_btn.pack(pady=5)
            self.buttons_frame.pack()
            messagebox.showinfo("Captured", "Image captured.")

    def retake_picture(self):
        self.image = None
        self.original_image = None
        self.result = None
        self.crop_mode = False
        self.canvas.delete("all")

        self.buttons_frame.pack_forget()
        self.retake_btn.pack_forget()
        self.capture_btn.pack(pady=5)

        self.previewing = True  # resume preview

    def show_static_image(self):
        img_resized = self.image.resize((400, 300))
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

    def rotate_image(self):
        if self.image:
            self.image = self.image.rotate(90, expand=True)
            self.show_static_image()

    def enable_crop_mode(self):
        if self.image:
            self.crop_mode = True
            self.draw_crop_box()

    def draw_crop_box(self):
        self.canvas.delete("crop")
        self.crop_box_coords = [50, 50, 350, 250]
        self.crop_box = self.canvas.create_rectangle(*self.crop_box_coords, outline='red', width=2, tags="crop")
        self.handles = {}
        for i, (x, y) in enumerate(self.get_handle_positions(*self.crop_box_coords)):
            self.handles[i] = self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill="blue", tags=("crop", f"handle{i}")
            )
        self.canvas.tag_bind("crop", "<ButtonPress-1>", self.start_drag_crop)
        self.canvas.tag_bind("crop", "<B1-Motion>", self.perform_drag_crop)
        self.canvas.tag_bind("crop", "<ButtonRelease-1>", self.finish_drag_crop)

    def get_handle_positions(self, x1, y1, x2, y2):
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def start_drag_crop(self, event):
        for i, handle_id in self.handles.items():
            x1, y1, x2, y2 = self.canvas.coords(handle_id)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.dragging = f"handle{i}"
                return
        if self.crop_box:
            x1, y1, x2, y2 = self.canvas.coords(self.crop_box)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.dragging = "box"
                self.drag_offset = (event.x - x1, event.y - y1)

    def perform_drag_crop(self, event):
        if not self.dragging or not self.crop_box:
            return
        coords = self.canvas.coords(self.crop_box)
        if self.dragging.startswith("handle"):
            i = int(self.dragging[-1])
            pos = list(coords)
            if i == 0:
                pos[0], pos[1] = event.x, event.y
            elif i == 1:
                pos[2], pos[1] = event.x, event.y
            elif i == 2:
                pos[2], pos[3] = event.x, event.y
            elif i == 3:
                pos[0], pos[3] = event.x, event.y
            self.canvas.coords(self.crop_box, *pos)
            for j, (hx, hy) in enumerate(self.get_handle_positions(*pos)):
                self.canvas.coords(self.handles[j], hx - 5, hy - 5, hx + 5, hy + 5)
        elif self.dragging == "box":
            dx, dy = event.x - self.drag_offset[0], event.y - self.drag_offset[1]
            x1, y1, x2, y2 = dx, dy, dx + (coords[2] - coords[0]), dy + (coords[3] - coords[1])
            self.canvas.coords(self.crop_box, x1, y1, x2, y2)
            for j, (hx, hy) in enumerate(self.get_handle_positions(x1, y1, x2, y2)):
                self.canvas.coords(self.handles[j], hx - 5, hy - 5, hx + 5, hy + 5)

    def finish_drag_crop(self, event):
        self.dragging = None

    def apply_crop(self):
        if not self.crop_mode or not self.image:
            return
        x1, y1, x2, y2 = [int(c) for c in self.canvas.coords(self.crop_box)]
        img_w, img_h = self.image.size
        scale_x = img_w / 400
        scale_y = img_h / 300
        crop_box = (
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y)
        )
        self.image = self.image.crop(crop_box)
        self.crop_mode = False
        self.canvas.delete("crop")
        self.show_static_image()

    def undo_crop(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.show_static_image()

    def analyze(self):
        if self.image:
            temp_path = "temp_to_analyze.jpg"
            self.image.save(temp_path)
            # self.result = analyze_image(temp_path)
            messagebox.showinfo("Result", f"Analysis: {self.result}")

    def save_to_csv(self):
        if self.result is not None:
            new_row = {"Image": captured_image_path, "Result": self.result}
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(output_csv, index=False)
            messagebox.showinfo("Saved", "Result saved to CSV!")

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
