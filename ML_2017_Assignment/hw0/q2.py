#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:14:27 2020
hw0 q2 of ntu ml 2017  
@author: cindy
"""
from PIL import Image, ImageChops

image = Image.open("lena.png")
image_modified = Image.open("lena_modified.png")
#image.show()
width, height = image.size
print(width, height)
width, height = image_modified.size
print(width, height)
#image_modified.show()

# compare the images pixels by pixels
# getpixel: Returns the pixel at x, y. The pixel is returned as a single
# putpixel: Modifies the pixel at x, y. The color is given as a single numerical value for single band images, and a tuple for multi-band images
w, h = image.size
for i in range(w):
    for j in range(h):
        if image.getpixel((i,j)) == image_modified.getpixel((i,j)):
            image_modified.putpixel((i,j),255)
image_modified.show()
image_modified.save("ans_two.png")

'''
diff = ImageChops.difference(image, image_modified)
if diff.getbbox():
    diff.show()
'''