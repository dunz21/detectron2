import cv2
import numpy as np

class PersonImageComparer:
    list_in = []
    list_out = []
    list_result = []
    list_banner_in = None
    offset_overlay_in = 0
    list_banner_out = None
    offset_overlay_out = 0
    max_width = 1920

    @staticmethod
    def find_match(image_in, image_out):
        # Assuming comparison is based on ID for simplicity
        return image_in.id == image_out.id

    @classmethod
    def process_person_image(cls, person_image):
        if person_image.list_images.__len__() > 0:
            if person_image.direction == "In":
                # Check if person_image is not already in cls.list_in
                if not any(p.id == person_image.id for p in cls.list_in):
                    cls.list_in.append(person_image)
                    cls.add_image_to_banner(person_image.list_images[0], person_image.direction)
                    print(f"Number of images of ID {person_image.id} {person_image.list_images.__len__()}")
            elif person_image.direction == "Out":
                # Check if person_image is not already in cls.list_out
                if not any(p.id == person_image.id for p in cls.list_out):
                    cls.list_out.append(person_image)
                    cls.add_image_to_banner(person_image.list_images[0], person_image.direction)
                    print(f"Number of images of ID {person_image.id} {person_image.list_images.__len__()}")
                    # cls.compare_and_process()
            
    @classmethod
    def compare_and_process(cls):
        if cls.list_out:
            for image_out in cls.list_out:
                for image_in in cls.list_in:
                    if cls.find_match(image_in, image_out):
                        cls.list_result.append((image_in, image_out))
                        cls.list_in.remove(image_in)
                        cls.list_out.remove(image_out)
                        return  # Exit after the first match is found
    @classmethod
    def add_image_to_banner(cls, path, direction):
        fixed_width, fixed_height = 50, 100
        img = cv2.imread(path)
        if img is not None:
            resized_img = cv2.resize(img, (fixed_width, fixed_height))

        if direction == "In":
            if cls.list_banner_in is None:
                cls.list_banner_in = resized_img
            else:
                if cls.list_banner_in.shape[1] + resized_img.shape[1] <= cls.max_width:
                    cls.list_banner_in = np.hstack((cls.list_banner_in, resized_img))
                else:
                    if cls.offset_overlay_in + resized_img.shape[1] >= cls.max_width:
                        cls.offset_overlay_in = 0
                    cls.list_banner_in[:, cls.offset_overlay_in:cls.offset_overlay_in+resized_img.shape[1], :] = resized_img
                    cls.offset_overlay_in += resized_img.shape[1]
        elif direction == "Out":
            if cls.list_banner_out is None:
                cls.list_banner_out = resized_img
            else:
                if cls.list_banner_out.shape[1] + resized_img.shape[1] <= cls.max_width:
                    cls.list_banner_out = np.hstack((cls.list_banner_out, resized_img))
                else:
                    if cls.offset_overlay_out + resized_img.shape[1] >= cls.max_width:
                        cls.offset_overlay_out = 0
                    cls.list_banner_out[:, cls.offset_overlay_out:cls.offset_overlay_out+resized_img.shape[1], :] = resized_img
                    cls.offset_overlay_out += resized_img.shape[1]


