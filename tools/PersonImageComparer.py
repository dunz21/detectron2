class PersonImageComparer:
    list_in = []
    list_out = []
    list_result = []

    @staticmethod
    def find_match(image_in, image_out):
        # Assuming comparison is based on ID for simplicity
        return image_in.id == image_out.id

    @classmethod
    def process_person_image(cls, person_image):
        if person_image.direction == "In":
            # Check if person_image is not already in cls.list_in
            if not any(p.id == person_image.id for p in cls.list_in):
                cls.list_in.append(person_image)
        elif person_image.direction == "Out":
            cls.list_out.append(person_image)
            cls.compare_and_process()
            
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