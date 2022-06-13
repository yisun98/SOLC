from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def tif2png(img):
    data = np.array(img)[:, :, :3].permute(2, 1, 0) # bgr-> rgb
    img = Image.fromarray(np.uint8(data[:, :]))
    return img

plt.figure(figsize=(20, 20))
for idx in range(4):
    # img_opt = Image.open(f"/home/sy/Seg/semantic_pytorch/result/input-opt/{idx}.tif")
    # img_sar = Image.open(f"/home/sy/Seg/semantic_pytorch/result/input-sar/{idx}.tif")
    img_opt = tif2png(Image.open(f"/home/sy/Seg/semantic_pytorch/result/input-opt/{idx}.png"))
    label = Image.open(f"/home/sy/Seg/semantic_pytorch/result/label/{idx}.png")
    pred = Image.open(f"/home/sy/Seg/semantic_pytorch/result/label/{idx}.png")
    mask = Image.open(f"/home/sy/Seg/semantic_pytorch/result/mask/{idx}.png")

    plt.subplot(4, 4, idx*4+1)
    plt.title('pic')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(4, 4, idx*4+2)
    plt.title('label')
    plt.imshow(label)
    plt.axis('off')

    plt.subplot(4, 4, idx*4+3)
    plt.title('predict')
    plt.imshow(pred)
    plt.axis('off')

    plt.subplot(4, 4, idx*4+4)
    plt.title('mask')
    plt.imshow(mask)
    plt.axis('off')
plt.savefig('result.png')
plt.show()