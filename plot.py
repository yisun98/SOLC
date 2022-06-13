from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random


def tif2png(img):
    data = np.array(img)[:, :, (2, 1, 0)]  # bgr-> rgb
    print(data.shape)
    img = Image.fromarray(np.uint8(data))

    return img


plt.figure(figsize=(20, 20))
for idx in range(4):

    epochs = [random.randint(0, 180) for i in range(4)]
    ids = [random.randint(0, 15) for i in range(4)]
    img = Image.open(f"/data/sy/experiments256/eight/deeplabv3_version_3/resnet50/05-24-09:14:22/predict/opt/img_batch_{epochs[idx]}_{ids[idx]}.jpg")
    label = Image.open(f"/data/sy/experiments256/eight/deeplabv3_version_3/resnet50/05-24-09:14:22/predict/gt/gt_{epochs[idx]}_{ids[idx]}.png")
    pred = Image.open(f"/data/sy/experiments256/eight/deeplabv3_version_3/resnet50/05-24-09:14:22/predict/predict/pre_{epochs[idx]}_{ids[idx]}.png")
    print(np.array(img).shape, np.array(label).shape)
    # mask = Image.blend(img, label, 0.3)

    plt.subplot(4, 4, idx * 4 + 1)
    plt.title('pic')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(4, 4, idx * 4 + 2)
    plt.title('label')
    plt.imshow(label)
    plt.axis('off')

    plt.subplot(4, 4, idx * 4 + 3)
    plt.title('predict')
    plt.imshow(pred)
    plt.axis('off')

plt.savefig('result.png')
plt.show()