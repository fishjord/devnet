#!/usr/bin/python

from dn import DN, Neuron
from dn_utils import write_image
from math import sqrt

import os
import sys
import numpy
import argparse
import math
import pickle
import struct
import math
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

params = {'backend': 'ps',
          'font.family' : 'cmr10',
          'font.serif' : 'Computer Modern Roman',
           'axes.labelsize': 8,
           'text.fontsize': 8,
           'legend.fontsize': 8,
           'xtick.labelsize': 6,
           'ytick.labelsize': 6,
           'text.usetex': True,
           'axes.linewidth' : 0.75
           }
matplotlib.rcParams.update(params)

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

    if m == 0:
        return numpy.array([0] * len(data), numpy.uint8)

    return numpy.array(((data - l) / (float(m))) * 255, numpy.uint8)

def main():
    if len(sys.argv) != 3:
        print >>sys.stderr, "USAGE: gen_images.py <net file> <stem>"
        sys.exit(1)

    stem = sys.argv[2]

    net = pickle.load(open(sys.argv[1], "rb"))
    xy_imgs = []
    yz_imgs = []
    zy_imgs = []

    sigma_imgs = []
    r_imgs = []

    yz_scale = 5

    ay_out = open("%s_ay.txt" % stem, "w")
    az_out = open("%s_az.txt" % stem, "w")
    y_ages = []
    z_ages = []

    for i in range(len(net.y.neurons)):
        y = net.y.neurons[i]
        sigma_imgs.append(rescale_img_color(y.sigmas[0]))
        print numpy.mean(y.sigmas[0])
        r_imgs.append(rescale_img_color(y.sigmas[0] / numpy.mean(y.sigmas[0])))
        xy_imgs.append(rescale_img_color(y.weights[0]))
        #zy_imgs.append(rescale_img_color(y.v_t))
        print >>ay_out, "%s\t%s" % (i, y.age)
        y_ages.append(y.age)

    n = int(sqrt(len(net.y.neurons)))

    for i in range(len(net.z.neurons)):
        z = net.z.neurons[i]
        weights = z.weights[0]
        new_weights = numpy.array([0.0] * (len(weights) * yz_scale * yz_scale))

        for i in range(n * yz_scale):
            for j in range(n * yz_scale):
                t = i / yz_scale * n + j / yz_scale
                new_weights[i * yz_scale * n + j] = weights[t]

        yz_imgs.append(rescale_img_color(new_weights))
        print >>az_out, "%s\t%s" % (i, z.age)
        z_ages.append(z.age)

    print r_imgs[0]
    print sigma_imgs[0]

    print "len(net.y)=%s, n=%s, yz_img=%s, len(yz_imgs)=%s, z=%s" % (len(net.y.neurons), n, len(yz_imgs), len(yz_imgs[0]), len(net.z.neurons))
    write_images(stem + "_%s_xy.pgm", expected_width, expected_height, xy_imgs)
    write_images(stem + "_%s_sigmas.pgm", expected_width, expected_height, sigma_imgs)
    write_images(stem + "_%s_rs.pgm", expected_width, expected_height, r_imgs)
    write_images(stem + "_%s_yz.pgm", n * yz_scale, n * yz_scale, yz_imgs)

    if len(zy_imgs) > 0:
        write_images(stem + "_%s_zy.pgm", len(net.z), n, zy_imgs)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(1.5,2)
    fig.subplots_adjust(bottom=0.18, left=.18)

    plt.bar(range(len(y_ages)), y_ages)
    #plt.title("Y Neuron Ages")
    plt.xlabel("Y Neuron")
    plt.ylabel("Firing Age")
    plt.show()
    plt.savefig("%s_y_ages.eps" % stem, dpi=226)
    plt.clf()

    plt.bar(range(len(z_ages)), z_ages)
    #plt.title("Z Neuron Ages")
    plt.xlabel("Z Neuron")
    plt.ylabel("Firing Age")
    plt.savefig("%s_z_ages.eps" % stem, dpi=226)

if __name__ == "__main__":
    main()
