import tkinter as tk
from gui import App
from ocr import analyze_image


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        csv_path = analyze_image("dataset")
        print(csv_path)
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
