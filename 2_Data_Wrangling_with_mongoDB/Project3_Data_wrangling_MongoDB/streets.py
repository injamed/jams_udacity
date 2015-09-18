__author__ = 'injamed'

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
import re

street_re = re.compile(r'^addr:street$')
end_word = re.compile(r'\s([^\s]*)$')

def what_streets(file_path):
    streets_dict = {}
    for _, element in ET.iterparse(file_path):
        for child in element:
            if child.tag == "tag":
                k_name = child.get("k", "")
                if re.match(street_re, k_name):
                    raw_street_name = child.get("v")
                    match = re.search(end_word, raw_street_name)
                    if match:
                        city_obj_name = match.group(1)
                        if city_obj_name not in streets_dict:
                            streets_dict[city_obj_name] = []
                        streets_dict[city_obj_name].append(raw_street_name)
    return streets_dict

if __name__ == "__main__":
    streets = what_streets("kyiv_ukraine_sample.osm")
    pass



