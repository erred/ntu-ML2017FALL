from PIL import Image
import numpy as np
import sys

image = Image.open(sys.argv[1])

width, height = image.size

image_new = Image.new("RGB", image.size, (0,0,0))

# arr = np.asarray(image, dtype=np.uint8)
for i in range(0, width):
    for j in range(0, height):
        pix = image.getpixel((i,j))
        image_new.putpixel((i,j), (int(pix[0]/2), int(pix[1]/2), int(pix[2]/2)))


image_new.save("Q2.jpg")
