from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageApp:
    def __init__(self, hala):
        self.hala = hala
        self.hala.title("application")

        self.canvas_original = Canvas(self.hala)
        self.canvas_modified = Canvas(self.hala)
        self.canvas_original.grid(row=1, column=0, padx=(10, 5), sticky="n")  
        self.canvas_modified.grid(row=1, column=1, padx=(5, 10), sticky="n")  

       
        self.label_original = Label(self.hala, text="Original Image")
        self.label_original.grid(row=2, column=0, padx=(10, 5), pady=5, sticky="n")

        self.label_modified = Label(self.hala, text="Modified Image")
        self.label_modified.grid(row=2, column=1, padx=(5, 10), pady=5, sticky="n")

        
        self.header_frame = Frame(self.hala)
        self.header_frame.grid(row=0, column=0, columnspan=2, pady=5)

       
        self.last_transformation = None

    
        self.normalized_gray_image = None

        
        filename = filedialog.askopenfilename(title="Open an image", filetypes=[('jpg files', '.jpg'),
                                                                                 ('png files', '.png'),
                                                                                 ('all files', '.*')])

        
        self.original_image = cv2.imread(filename)
        self.display_original_image(self.original_image)

    
        buttons = [
            ("HSV", self.apply_hsv),
            ("Gray", self.apply_gray),
            ("Binary", self.apply_binary),
            ("Invert Colors", self.invert_colors),
            ("Original Histogram", self.show_original_histogram),
            ("Normalize", self.apply_normalize_gray),
            ("Hist Normalize", self.show_histogram),
            ("Equalize", self.apply_equalization),
            ("Hist Equalize", self.show_equalized_histogram),
            ("Average Filter", self.apply_filter),
            ("Laplacian Filter", self.apply_laplacian_filter),
            ("Emboss Filter", self.apply_emboss_filter),
            ("Adaptive Filter", self.apply_adaptive_filter),
            ("Restore", self.restore_original),
            ("Save", self.save_image),
        ]

        for i, (text, command) in enumerate(buttons):
            button = Button(self.header_frame, text=text, command=lambda cmd=command, txt=text: self.apply_and_update(cmd, txt))
            button.grid(row=0, column=i, padx=5, sticky="n")

    def apply_and_update(self, command, text):
        modified_image = command()
        self.last_transformation = modified_image
        self.label_modified.config(text=f"Modified Image in {text}")
        self.display_modified_image(modified_image)

    def apply_hsv(self):
        
        return cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

    def apply_gray(self):
       
        return cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def apply_binary(self):
        
        _, binary_image = cv2.threshold(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 128, 255,
                                        cv2.THRESH_BINARY)
        return binary_image

    def apply_normalization(self, gray_image):
       
        normalized_gray_image = self.normalize_gray_image(gray_image)
        return normalized_gray_image

    def apply_equalization(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        return equalized_image

    def show_histogram(self):
        plt.hist(self.normalized_gray_image.flatten(), 256, [0, 256], color='g', alpha=0.7, label='Normalized')
        plt.legend(loc='upper right')
        plt.show()

    def show_equalized_histogram(self):
        equalized_gray_image = cv2.equalizeHist(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY))
        plt.hist(equalized_gray_image.flatten(), 256, [0, 256], color='b', label='Equalized')
        plt.legend(loc='upper right')
        plt.show()

    def show_original_histogram(self):
        plt.hist(self.original_image.flatten(), 256, [0, 256], color='r', label='Original')
        plt.legend(loc='upper right')
        plt.show()

    def apply_filter(self):
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(self.original_image, -1, kernel)

    def restore_original(self):
        return self.original_image

    def save_image(self):
       
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"),
                                                                                   ("JPEG files", "*.jpg"),
                                                                                   ("BMP files", "*.bmp")])
        if file_path:

            if not file_path.lower().endswith((".png", ".jpg")):
                file_path += ".png"

            cv2.imwrite(file_path, self.last_transformation)

    def invert_colors(self):
        
        return cv2.bitwise_not(self.original_image)

    def display_original_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas_original.config(height=height, width=width)
        self.canvas_original.create_image(0, 0, anchor=NW, image=photo)
        self.canvas_original.photo = photo

    def display_modified_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas_modified.config(height=height, width=width)
        self.canvas_modified.create_image(0, 0, anchor=NW, image=photo)
        self.canvas_modified.photo = photo

    def normalize_gray_image(self, gray_image):
        
        image_float = gray_image.astype(np.float32)

       
        normalized_gray_image = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)

        return normalized_gray_image.astype(np.uint8)

    def apply_laplacian_filter(self):
       
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_image = np.uint8(np.abs(laplacian_image))
        return laplacian_image

    def apply_emboss_filter(self):
        
        emboss_filter = np.array([[-2, -1, 0],
                                  [-1, 1, 1],
                                  [0, 1, 2]])
        emboss_image = cv2.filter2D(self.original_image, -1, emboss_filter)
        return emboss_image

    def apply_adaptive_filter(self):
       
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        adaptive_filtered_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        return adaptive_filtered_image                                   #cv2.adaptative_thresh_mean_c (type de seuillage ici est la moyenne pondérée du voisinage.)

    def apply_normalize_gray(self):
        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.normalized_gray_image = self.normalize_gray_image(gray_image)

        return self.normalized_gray_image

if __name__ == "__main__":
    root = Tk()
    app = ImageApp(root)
    root.mainloop()
