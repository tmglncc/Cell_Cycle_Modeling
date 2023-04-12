import sys
import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
from shutil import copyfile
import pathlib

def create_xml_file(xml_file_in, xml_file_out, parameters):
	if not os.path.exists(xml_file_out):
		print("Creating " + str(pathlib.Path(xml_file_out).parent))
		os.makedirs(pathlib.Path(xml_file_out).parent)
	copyfile(xml_file_in, xml_file_out)
	tree = ET.parse(xml_file_out)
	xml_root = tree.getroot()
	for key in parameters.keys():
		val = parameters[key]
		print(key + " => " + val)
		if ('.' in key):
			k = key.split('.')
			uep = xml_root
			for idx in range(len(k)):
				uep = uep.find('.//' + k[idx])  # unique entry point (uep) into xml
			uep.text = val
		else:
			if (key == 'folder' and not os.path.exists(val)):
				print('Creating ' + val)
				os.makedirs(val)

			xml_root.find('.//' + key).text = val
	tree.write(xml_file_out)

def create_config_folders(num_cells_list, output_folder, config_filename):
	for num_cells in num_cells_list:
		folder_name = output_folder+"output_NC"+str("%03d"%num_cells)+'/'
		parameters = {'folder': folder_name, 'number_of_cells': str(num_cells)}
		create_xml_file('config/'+config_filename, folder_name+config_filename, parameters)

def run_model(num_cells_list, output_folder, config_filename):
	for num_cells in num_cells_list:
		print('Running number of cells=' + str(num_cells))
		folder_name = output_folder+"output_NC"+str("%03d"%num_cells)+'/'

		# Write input for simulation & execute
		calling_model = ['./project', folder_name+config_filename]
		cache = subprocess.run(calling_model, universal_newlines=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		if (cache.returncode != 0):
			print("Model output error! Returned: "+ str(cache.returncode))
			os._exit(1)

num_cells_list = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

# output_folder = "output_Live/"
output_folder = "output_Ki67_Basic/"
# output_folder = "output_Ki67_Advanced/"
# output_folder = "output_Flow_Cytometry/"
# output_folder = "output_Separated_Flow_Cytometry/"

# config_filename = "PhysiCell_settings_Live_without_o2_apop_nec.xml"
config_filename = "PhysiCell_settings_Ki67_Basic_without_o2_apop_nec.xml"
# config_filename = "PhysiCell_settings_Ki67_Advanced_without_o2_apop_nec.xml"
# config_filename = "PhysiCell_settings_Flow_Cytometry_without_o2_apop_nec.xml"
# config_filename = "PhysiCell_settings_Separated_Flow_Cytometry_without_o2_apop_nec.xml"

create_config_folders(num_cells_list, output_folder, config_filename)
run_model(num_cells_list, output_folder, config_filename)