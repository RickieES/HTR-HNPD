#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:11:28 2025

@author: rpalomares
"""

import argparse
import cv2
import numpy as np
from os import makedirs, walk
from os.path import exists, join, splitext
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

def build_file_list(input_dir):
    """
    Builds the list of XML files in a directory, excepting METS.xml, created
    by eScriptorium on ALTO format export

    Parameters
    ----------
    input_dir : string
        The path where XML files live.

    Returns
    -------
    list of strings
        The ALTO or PAGE XML file collection.

    """

    file_col = []
    for root, dirs, files in walk(input_dir):
         for file in files:
             if (file.lower() == 'mets.xml'):
                 continue

             ext_filename = file[file.rfind('.'):]
             if ((ext_filename.lower() in ['.xml'])
                 and (file.lower != "mets.xml")):
                 full_input_filename = join(root, file)
                 file_col.append({'input': full_input_filename})
    file_col.sort(key = lambda filename: filename['input'])
    return file_col


def parse_xml(input_xml):
    """
    Parses an ALTO or PAGE XML

    Parameters
    ----------
    input_xml : string
        pathname to XML file.

    Returns
    -------
    dictionary
        dictionary with the extracted information.

    """

    xml_tree = ET.parse(input_xml)
    xml_root = xml_tree.getroot()

    parsed_xml = {}
    if xml_root.tag.endswith('alto'):
        parsed_xml = parse_ALTO_xml(xml_root)
    elif xml_root.tag.endswith('PcGts'):
        parsed_xml = parse_PAGE_xml(xml_root)
    else:
        parsed_xml = {}

    return parsed_xml


def parse_ALTO_xml(xml_root):
    """
    Extracts relevant information from an ALTO XML file, which is
    - Name of image file
    - Collection of text line regions, including width, height, associated
      label and polygon defining it

    Parameters
    ----------
    xml_root : xml.etree.Element
        the XML root element.

    Returns
    -------
    img_alto_dict : dictionary
        dictionary with the extracted information.

    """

    img_alto_dict = {}
    # Extracts image file pathname
    img_alto_dict['img_filename'] = xml_root.find(
        './{*}Description/{*}sourceImageInformation/{*}fileName').text

    # Extracts each line region
    img_alto_dict['text_regions_col'] = []
    for text_line in xml_root.findall(
            './{*}Layout/{*}Page/{*}PrintSpace/{*}TextBlock/{*}TextLine'):
        polygon_elem = text_line.find('./{*}Shape/{*}Polygon')
        string_elem = text_line.find('./{*}String')

        imgregion_txt_dict = {}
        imgregion_txt_dict['text'] = string_elem.attrib['CONTENT']
        imgregion_txt_dict['img_width'] = text_line.attrib['WIDTH']
        imgregion_txt_dict['img_height'] = text_line.attrib['HEIGHT']

        # ALTO format contains each coord separated by a single space, every
        # two forming a point (so, instead of x,y x,y... we have x y x y)
        # whereas PAGE format contains each point in x,y which is better for
        # further processing, so we standardize on it
        p = polygon_elem.attrib['POINTS']
        pol = ""
        point = ""
        for coord in p.split(' '):
            if (point == ""):
                point = coord
            else:
                pol = pol + point + "," + coord + " "
                point = ""

        imgregion_txt_dict['polygon'] = pol.strip()
        img_alto_dict['text_regions_col'].append(imgregion_txt_dict)

    return img_alto_dict


def parse_PAGE_xml(xml_root):
    """
    Extracts relevant information from a PAGE XML file, which is
    - Name of image file
    - Collection of text line regions, including width, height, associated
      label and polygon defining it

    Parameters
    ----------
    xml_root : xml.etree.Element
        the XML root element.

    Returns
    -------
    img_page_dict : dictionary
        dictionary with the extracted information.

    """

    img_page_dict = {}
    # Extracts image file pathname
    img_page_dict['img_filename'] = xml_root.find(
        './{*}Page').attrib['imageFilename']

    # Gets the regions ordered by their ids
    refRegions = []
    for refRegion in xml_root.findall(
            './{*}Page/{*}ReadingOrder/{*}OrderedGroup/{*}RegionRefIndexed'):
        refRegions.append(refRegion)
    refRegions.sort(key = lambda refReg: refReg.attrib['index'])

    # Extracts each line region
    img_page_dict['text_regions_col'] = []
    for refRegion in refRegions:
        region = xml_root.find("./{*}Page/{*}TextRegion[@id='"
                               + refRegion.attrib['regionRef'] + "']")
        for text_line in region.findall(
                './{*}TextLine'):
            polygon_elem = text_line.find('./{*}Coords')
            string_elem = text_line.find('./{*}TextEquiv/{*}Unicode')

            imgregion_txt_dict = {}
            imgregion_txt_dict['polygon'] = polygon_elem.attrib['points']

            if string_elem.text == None:
                imgregion_txt_dict['text'] = ""
            else:
                imgregion_txt_dict['text'] = string_elem.text

            # PAGE format does not include width and height attributes, so
            # we have to compute them from the points collection
            min_x = 10000000
            min_y = 10000000
            max_x = 0
            max_y = 0
            for point in imgregion_txt_dict['polygon'].split(' '):
                coords = point.split(',')
                min_x = int(coords[0]) if int(coords[0]) < min_x else min_x
                min_y = int(coords[1]) if int(coords[1]) < min_y else min_y
                max_x = int(coords[0]) if int(coords[0]) > max_x else max_x
                max_y = int(coords[1]) if int(coords[1]) > max_y else max_y
            w = (max_x - min_x) + 1
            h = (max_y - min_y) + 1
            imgregion_txt_dict['img_width'] = w
            imgregion_txt_dict['img_height'] = h

            img_page_dict['text_regions_col'].append(imgregion_txt_dict)

    return img_page_dict


def extract_image_region(img, polygon, img_region_filename,
                         adjusted_height_mean):
    """
    Extracts each line region of an image, resizes it and saves it on disk

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    polygon : TYPE
        DESCRIPTION.
    img_region_filename : TYPE
        DESCRIPTION.
    adjusted_height_mean : TYPE
        DESCRIPTION.

    Returns
    -------
    adjusted_width : TYPE
        DESCRIPTION.

    """


    #
    # x 0, y 0 coordinates are located top-left of image
    #
    min_x = 10000000
    min_y = 10000000
    max_x = 0
    max_y = 0
    tmp_coord_list = []

    # Find the enclosing rectangle of the region
    for point in polygon.split(' '):
        coords = point.split(',')
        min_x = int(coords[0]) if int(coords[0]) < min_x else min_x
        min_y = int(coords[1]) if int(coords[1]) < min_y else min_y
        max_x = int(coords[0]) if int(coords[0]) > max_x else max_x
        max_y = int(coords[1]) if int(coords[1]) > max_y else max_y
        tmp_coord_list.append([int(coords[0]), int(coords[1])])

    prev_coordinate = []
    img_coord_list = []
    for coordinate in tmp_coord_list:
        if len(prev_coordinate) > 0:
            too_similar = ((abs(prev_coordinate[0] - coordinate[0]) < 2) and
                          (abs(prev_coordinate[1] - coordinate[1]) < 2))
            if (too_similar):
                # Don't copy the element, as it is too close to the previous one
                continue
        else:
            prev_coordinate.extend([0,0])

        img_coord_list.append(coordinate)
        prev_coordinate[0] = coordinate[0]
        prev_coordinate[1] = coordinate[1]

    # Create a black mask of full image size
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    # Create a white mask with the polygon shape
    cv2.fillPoly(mask, [np.array(img_coord_list)], 1)
    mask = mask > 0 # To convert to Boolean

    # Create another rectangular black mask enclosing just the polygon
    out = np.zeros_like(img)
    # And get the
    out[mask] = img[mask]
    out2 = out[min_y:max_y, min_x:max_x]
    adjusted_width = round(out2.shape[1] * adjusted_height_mean / out2.shape[0])
    result = cv2.resize(out2, (adjusted_width, adjusted_height_mean),
                        interpolation = cv2.INTER_AREA)
    cv2.imwrite(img_region_filename, result)

    return adjusted_width



#
# Main process
#
if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-I", "--input_dir", required=True,
    	help="path to folder containing an ALTO or PAGE dataset (XML + images")
    ap.add_argument("-O", "--output_dir", required=False, default="lines",
    	help="path to folder containing output images \n"
        + "(defaults to a subdir lines inside input_dir)")
    ap.add_argument("-R", "--resize", required=False,
        help="resized height in pixels (if not provided, a sensible mean height will be used)")
    args = vars(ap.parse_args())

    file_col = []
    if (exists(args['input_dir'])):
        file_col = build_file_list(args['input_dir'])
    else:
        print('Input directory not found')
        sys.exit(-1)

    if (args['output_dir'] == "lines"):
        args['output_dir'] = join(args['input_dir'], args['output_dir'])

    if (not (exists(args['output_dir']))):
        makedirs(args['output_dir'])

    if (args['resize'] != None):
        adjusted_height_mean = int(args['resize'])
    else:
       # Calculate the height mean to select a sensible resizing height value
        img_size = []
        for fo in file_col:
            imgregion_txt_col = parse_xml(fo['input'])
            for region_col in imgregion_txt_col['text_regions_col']:
                img_size.append([int(region_col['img_width']),
                                 int(region_col['img_height'])])
    
        dim_mean = np.mean(img_size, axis = 0)
        print(f"Mean dimensions: width: {dim_mean[0]} - height: {dim_mean[1]}")
        height_mean = dim_mean[1]
        adjusted_height_mean = 1
        while (adjusted_height_mean < height_mean):
            adjusted_height_mean = adjusted_height_mean * 2
        adjusted_height_mean = adjusted_height_mean // 2
        print(f"Height mean: {height_mean}")
        print(f"Adjusted height mean: {adjusted_height_mean}\n")

    adjusted_max_width = 0
    with open(args['output_dir'] + '/dataset.txt', 'w',
              encoding='utf-8') as dtxt:
        for fo in tqdm(file_col):
            imgregion_txt_col = parse_xml(fo['input'])

            image = cv2.imread(args['input_dir'] + '/' +
                               imgregion_txt_col['img_filename'], cv2.IMREAD_GRAYSCALE)
            img_filename_seed = args['output_dir'] + '/' + \
                                splitext(imgregion_txt_col['img_filename'])[0] + \
                                "_"
            region_idx = 1

            for region_col in imgregion_txt_col['text_regions_col']:
                region = region_col['polygon']
                img_region_filename = img_filename_seed + format(region_idx,'0=2d') + '.png'
                adjusted_width = extract_image_region(image, region,
                                                           img_region_filename,
                                                           adjusted_height_mean)
                dtxt.write(img_region_filename + '|' + region_col['text'] + '\n')
                adjusted_max_width = max(adjusted_width, adjusted_max_width)
                region_idx = region_idx + 1

    print(f"\n\nAdjusted max width: {adjusted_max_width}\n")

    sys.exit()
