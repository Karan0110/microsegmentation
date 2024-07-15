# Had to implement by hand - for some reason it is not supported on the HPC environment

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

class ElasticTransform:
    def __init__(self, alpha, sigma, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state

    def __call__(self, img):
        image = np.array(img)

        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state = self.random_state

        shape = image.shape
        dx = random_state.rand(*shape[:2]) * 2 - 1
        dy = random_state.rand(*shape[:2]) * 2 - 1
        dx = cv2.GaussianBlur(dx, (17, 17), self.sigma) * self.alpha #type: ignore
        dy = cv2.GaussianBlur(dy, (17, 17), self.sigma) * self.alpha #type: ignore

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(distorted_image)

