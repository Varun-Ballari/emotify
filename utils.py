import base64
import binascii
import threading

from time import sleep
from io import BytesIO
from PIL import Image


def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def flip_image(image_array):
    img = Image.fromarray(image_array)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = pil_image_to_base64(img)
    return binascii.a2b_base64(img)


class Queue(object):
    def __init__(self, model):
        self.to_process = []
        self.to_output = []
        self.model = model

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string.
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)
        output_img, emotion, user = self.model.predict_emotion(input_img)
        output_img = flip_image(output_img)
        self.to_output.append((output_img, emotion, user))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def dequeue(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)
