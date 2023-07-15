import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)    # 5
print(img.shape)    # (784,)
img = img.reshape(28, 28)
print(img.shape)    # (28, 28)

img_show(img)