#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import sys
import image_preprocessing_functions as ipf
from os import makedirs, walk, sep
from os.path import dirname, exists, join
from tqdm import tqdm

def build_file_list(input_dir, output_dir):
    """
    Build the file list of images to process and the output pathname

    Parameters
    ----------
    input_dir : string
        the input directory, where the original image files live.
    output_dir : string
        the output directory, where the modified image files will be saved.

    Returns
    -------
    dictionary
        a dictionary with the input and output full pathnames.

    """
    
    file_col = []

    for root, dirs, files in walk(input_dir):
         for file in files:
             ext_filename = file[file.rfind('.'):]
             if (ext_filename.lower() in ['.png', '.jpg', '.jpeg']):
                 output_filename = file[:file.rfind('.')] \
                     + '_output' + ext_filename
                 full_input_filename = join(root, file)
                 full_output_filename = join(root, output_filename)
                 full_output_filename = full_output_filename.replace(
                     input_dir, output_dir, 1)
                 file_col.append({'input': full_input_filename,
                                  'output': full_output_filename})
    return file_col


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help="path to single input image")
ap.add_argument("-I", "--input_dir", required=False,
	help="path to folder containing input images")
ap.add_argument("-o", "--output", required=False,
	help="path to single output image")
ap.add_argument("-O", "--output_dir", required=False,
	help="path to folder containing output images")
ap.add_argument("-c", "--clahe", action="store_true", required=False,
	help="use CLAHE to enhance contrast and brightness (slower and better)")
ap.add_argument("-b", "--binarize", action="store_true", required=False,
	help="binarize image")
ap.add_argument("-e", "--erode_dilate", action="store_true", required=False,
	help="apply erosion and dilation using a 3x3 kernel size")

args = vars(ap.parse_args())

# If neither -i nor -I provided, exit with error message
if ((args['input'] == None) and (args['input_dir'] == None)):
    print("Either -i or -I must be provided")
    ap.print_help()
    sys.exit(-1)

# If both -i and -I provided, or both -o and -O, exit with error message
if (((args['input'] != None) and (args['input_dir'] != None)) or
    ((args['output'] != None) and (args['output_dir'] != None))):
    print("Can't use both -i and -I, nor -o and -O")
    ap.print_help()
    sys.exit(-2)

# If -i provided but not -o, exit with error message
if ((args['input'] != None) and (args['output'] == None)):
    print("If -i is provided, then -o must also be provided")
    ap.print_help()
    sys.exit(-3)

# If -I provided but not -O, exit with error message
if ((args['input_dir'] != None) and (args['output_dir'] == None)):
    print("If -I is provided, then -O must also be provided")
    ap.print_help()
    sys.exit(-4)


# File collection to process, a row per file
file_col = []

# If -i provided, populate file_col with just that file
if (args['input'] != None):
    if (exists(args['input'])):
        file_col.append({'input': args['input'], 'output': args['output']})
    else:
        print('Input file not found!')
# Else (then -I have been provided), build the full list
else:
    if (exists(args['input_dir'])):
        if (args['output_dir'][-1] != sep):
            args['output_dir'] = args['output_dir'] + sep
        file_col = build_file_list(args['input_dir'], args['output_dir'])
        if (not (exists(args['output_dir']))):
            makedirs(args['output_dir'])
    else:
        print('Input directory not found')

use_CLAHE = args['clahe']
binarize_image = args['binarize']
erode_dilate = args['erode_dilate']

for fo in tqdm(file_col):
    # Load an image in grayscale
    image = cv2.imread(fo['input'], cv2.IMREAD_GRAYSCALE)

    image = ipf.enhance_contrast(image, use_CLAHE)

    image = ipf.denoise(image)

    if erode_dilate:
        image = ipf.erode_dilate(image)

    if binarize_image:
        image = ipf.binarize(image)

    if (not exists(dirname(fo['output']))):
        makedirs(dirname(fo['output']))

    cv2.imwrite(fo['output'], image)

sys.exit()
