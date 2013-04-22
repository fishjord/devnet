#!/usr/bin/python

from dn_utils import write_image
from dn_utils import read_face

import os
import sys
import numpy
import argparse
import math
import pickle
import struct
import math
from collections import namedtuple

page_width_72ppi = int(7 * 72)
page_height_72ppi = int(8 * 72)
expected_width = 64
expected_height = 88

def write_images(fname, img_width, img_height, all_imgs):
    cols_per_line = page_width_72ppi / (img_width + 1)
    rows_per_page = page_height_72ppi / (img_height + 1)

    white_img = numpy.ones(img_width * img_height) * 255

    w = (img_width + 1) * cols_per_line - 1
    h = (img_height + 1) * rows_per_page - 1

    imgs_per_page = rows_per_page * cols_per_line
    pages = int(math.ceil(len(all_imgs) / float(imgs_per_page)))

    for page in range(pages):
        out = open(fname % page, "wb")
        imgs = all_imgs[imgs_per_page * page : imgs_per_page * (page + 1)]
        
        out.write("P5\n%s  %s  255\n" % (w, h))

        for row in range(rows_per_page):
            for y in range(img_height):
                for col in range(cols_per_line):
                    i = cols_per_line * row + col
                    if i < len(imgs):
                        img = imgs[i]
                    else:
                        img = white_img

                    offset = y * img_width
                    if offset == len(img):
                        break

                    for i in range(img_width):
                        #print offset + i
                        out.write(struct.pack(">B", img[offset + i]))

                    if col + 1 != cols_per_line:
                        out.write(struct.pack(">B", 255))

            if row + 1 != rows_per_page:
                for i in range(w):
                    out.write(struct.pack(">B", 255))

        out.close()

def rescale_img_color(data):
    if data == None:
        return None

    l = min(data)
    u = max(data)
    m = (u - l)

    return numpy.array(((data - l) / (float(m))) * 255, numpy.uint8)

def main():
    faces = []
    for f in sys.argv[1:]:
        faces.append(read_face(f))
    write_images("combined_%s.pgm", expected_width, expected_height, faces)

if __name__ == "__main__":
    main()
