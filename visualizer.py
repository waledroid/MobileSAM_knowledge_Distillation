# import os
# import numpy as np
# from PIL import Image, ImageTk, ImageDraw
# import tkinter as tk
# from nanosam.utils.predictor import Predictor
# import argparse

# class ImageSegmenter:
#     def __init__(self, root, image_encoder, mask_decoder, image_path):
#         self.root = root
#         self.image_encoder = image_encoder
#         self.mask_decoder = mask_decoder
        
#         self.canvas_width = 800
#         self.canvas_height = 600
        
#         self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, cursor="cross")
#         self.canvas.pack(fill=tk.BOTH, expand=True)
        
#         self.original_image = Image.open(image_path)
#         self.image = self.resize_image_to_fit_canvas(self.original_image, self.canvas_width, self.canvas_height)
#         self.tk_image = ImageTk.PhotoImage(self.image)
        
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
#         self.rect = None
#         self.start_x = None
#         self.start_y = None
#         self.end_x = None
#         self.end_y = None
        
#         self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#         self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
#         self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

#         self.predictor = None
#         self.mask_image = None
#         self.overlay_image = None
#         self.overlay_image_id = None  # To keep track of the overlay image ID on canvas

#         # Display model information
#         self.display_model_info()

#     def resize_image_to_fit_canvas(self, image, canvas_width, canvas_height):
#         img_width, img_height = image.size
#         aspect_ratio = img_width / img_height
#         if img_width > img_height:
#             new_width = canvas_width
#             new_height = int(new_width / aspect_ratio)
#         else:
#             new_height = canvas_height
#             new_width = int(new_height * aspect_ratio)
        
#         return image.resize((new_width, new_height), Image.LANCZOS)

#     def on_button_press(self, event):
#         # Reload the original image on canvas
#         self.canvas.delete("all")
#         self.tk_image = ImageTk.PhotoImage(self.image)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
#         # Reset predictor
#         self.reset_predictor()

#         self.start_x = event.x
#         self.start_y = event.y
#         if self.rect:
#             self.canvas.delete(self.rect)
#         self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

#     def on_mouse_drag(self, event):
#         self.end_x = event.x
#         self.end_y = event.y
#         self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

#     def on_button_release(self, event):
#         if self.rect:
#             self.canvas.delete(self.rect)
#         self.rect = None
        
#         if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
#             x0, y0 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
#             x1, y1 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            
#             bbox = [x0, y0, x1, y1]
#             points = np.array([
#                 [bbox[0], bbox[1]],
#                 [bbox[2], bbox[3]]
#             ])
#             point_labels = np.array([2, 3])
            
#             # Reset predictor and perform prediction
#             self.reset_predictor()
#             mask, _, _ = self.predictor.predict(points, point_labels)
#             mask = (mask[0, 0] > 0).detach().cpu().numpy()
            
#             # Convert mask to a green Image for displaying
#             self.mask_image = self.create_green_mask(mask)
#             self.mask_image = self.mask_image.resize(self.image.size, resample=Image.NEAREST)
            
#             # Overlay mask on the original image
#             self.overlay_image = Image.blend(self.image.convert('RGBA'), self.mask_image.convert('RGBA'), alpha=0.5)
            
#             # Update the canvas with the new overlay image
#             self.update_canvas()

#     def reset_predictor(self):
#         # Close previous predictor if exists
#         if self.predictor:
#             del self.predictor
        
#         # Create new predictor instance
#         self.predictor = Predictor(self.image_encoder, self.mask_decoder)
#         self.predictor.set_image(self.image)

#     def update_canvas(self):
#         # Clear previous overlay image if exists
#         if self.overlay_image_id:
#             self.canvas.delete(self.overlay_image_id)
        
#         # Display original image with overlay
#         self.tk_overlay_image = ImageTk.PhotoImage(self.overlay_image)
#         self.overlay_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_overlay_image)
        
#         # Update rectangle if exists
#         if self.rect:
#             self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

#     def create_green_mask(self, mask):
#         """Creates a green mask image from the given boolean mask array."""
#         mask_image = Image.new("RGBA", (mask.shape[1], mask.shape[0]), (0, 0, 0, 0))
#         draw = ImageDraw.Draw(mask_image)
#         for y in range(mask.shape[0]):
#             for x in range(mask.shape[1]):
#                 if mask[y, x]:
#                     draw.point((x, y), fill=(0, 255, 0, 128))  # Green color with transparency
#         return mask_image

#     def display_model_info(self):
#         # Extract filenames from paths
#         encoder_filename = os.path.basename(self.image_encoder)
#         decoder_filename = os.path.basename(self.mask_decoder)

#         # Create labels for displaying model information
#         encoder_label = tk.Label(self.root, text=f"Image Encoder: {encoder_filename}", bg="purple", fg="white", anchor="w")
#         encoder_label.place(x=10, y=10)

#         decoder_label = tk.Label(self.root, text=f"Mask Decoder: {decoder_filename}", bg="purple", fg="white", anchor="w")
#         decoder_label.place(x=10, y=30)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Interactive Image Segmenter")
#     parser.add_argument("--image_path", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/waste.jpg", help="Path to the input image")
#     parser.add_argument("--image_encoder", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/models/mobilevit_s/mobilevit_s_image_encoder.engine", help="Path to the image encoder model file")
#     parser.add_argument("--mask_decoder", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/mobile_sam_mask_decoder.engine", help="Path to the mask decoder model file")
#     args = parser.parse_args()

#     root = tk.Tk()
#     root.title("WALE_NANO")  # Set window title
#     root.geometry("800x600")
#     root.resizable(False, False)  # Disable window resizing

#     segmenter = ImageSegmenter(root, args.image_encoder, args.mask_decoder, args.image_path)
#     root.mainloop()


import os
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import torch  # Add this line
from nanosam.utils.predictor import Predictor
import argparse

class ImageSegmenter:
    def __init__(self, root, image_encoder, mask_decoder, image_path):
        self.root = root
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        
        self.canvas_width = 800
        self.canvas_height = 600
        
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.original_image = Image.open(image_path)
        self.image = self.resize_image_to_fit_canvas(self.original_image, self.canvas_width, self.canvas_height)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.predictor = None
        self.mask_image = None
        self.overlay_image = None
        self.overlay_image_id = None  # To keep track of the overlay image ID on canvas

        # Display model information
        self.display_model_info()

    def resize_image_to_fit_canvas(self, image, canvas_width, canvas_height):
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        if img_width > img_height:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        
        return image.resize((new_width, new_height), Image.LANCZOS)

    def on_button_press(self, event):
        # Reload the original image on canvas
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Reset predictor
        self.reset_predictor()

        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

    def on_button_release(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = None
        
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
            x0, y0 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
            x1, y1 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            
            bbox = [x0, y0, x1, y1]
            points = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[3]]
            ])
            point_labels = np.array([2, 3])
            
            # Reset predictor and perform prediction
            self.reset_predictor()
            mask, class_names, _ = self.predictor.predict(points, point_labels)
            mask = (mask[0, 0] > 0).detach().cpu().numpy()
            
            # Convert mask to a green Image for displaying
            self.mask_image = self.create_green_mask(mask)
            self.mask_image = self.mask_image.resize(self.image.size, resample=Image.NEAREST)
            
            # Overlay mask on the original image
            self.overlay_image = Image.blend(self.image.convert('RGBA'), self.mask_image.convert('RGBA'), alpha=0.5)
            
            # Convert class_names to a list of strings if it's a tensor
            if isinstance(class_names, torch.Tensor):
                class_names = class_names.tolist()
                class_names = [str(name) for name in class_names]
            
            # Draw class names on the overlay image
            self.overlay_image = self.draw_class_names(self.overlay_image, class_names, bbox)
            
            # Update the canvas with the new overlay image
            self.update_canvas()

    def reset_predictor(self):
        # Close previous predictor if exists
        if self.predictor:
            del self.predictor
        
        # Create new predictor instance
        self.predictor = Predictor(self.image_encoder, self.mask_decoder)
        self.predictor.set_image(self.image)

    def update_canvas(self):
        # Clear previous overlay image if exists
        if self.overlay_image_id:
            self.canvas.delete(self.overlay_image_id)
        
        # Display original image with overlay
        self.tk_overlay_image = ImageTk.PhotoImage(self.overlay_image)
        self.overlay_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_overlay_image)
        
        # Update rectangle if exists
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)

    def create_green_mask(self, mask):
        """Creates a green mask image from the given boolean mask array."""
        mask_image = Image.new("RGBA", (mask.shape[1], mask.shape[0]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_image)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    draw.point((x, y), fill=(0, 255, 0, 128))  # Green color with transparency
        return mask_image

    def draw_class_names(self, image, class_names, bbox):
        """Draws class names on the image."""
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_position = (bbox[0], bbox[1] - 10)
        for class_name in class_names:
            draw.text(text_position, class_name, fill="red", font=font)
            text_position = (text_position[0], text_position[1] + 15)
        return image

    def display_model_info(self):
        # Extract filenames from paths
        encoder_filename = os.path.basename(self.image_encoder)
        decoder_filename = os.path.basename(self.mask_decoder)

        # Create labels for displaying model information
        encoder_label = tk.Label(self.root, text=f"Image Encoder: {encoder_filename}", bg="purple", fg="white", anchor="w")
        encoder_label.place(x=10, y=10)

        decoder_label = tk.Label(self.root, text=f"Mask Decoder: {decoder_filename}", bg="purple", fg="white", anchor="w")
        decoder_label.place(x=10, y=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Image Segmenter")
    parser.add_argument("--image_path", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/waste.jpg", help="Path to the input image")
    parser.add_argument("--image_encoder", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/models/mobilevit_s/mobilevit_s_image_encoder.engine", help="Path to the image encoder model file")
    parser.add_argument("--mask_decoder", type=str, default="/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/mobile_sam_mask_decoder.engine", help="Path to the mask decoder model file")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("WALE_NANO")  # Set window title
    root.geometry("800x600")
    root.resizable(False, False)  # Disable window resizing

    segmenter = ImageSegmenter(root, args.image_encoder, args.mask_decoder, args.image_path)
    root.mainloop()

