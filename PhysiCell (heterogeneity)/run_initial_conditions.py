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

def create_config_folders(tumor_radius_list, output_folder, config_filename):
	for tumor_radius in tumor_radius_list:
		folder_name = output_folder+"output_TR"+str("%03d"%tumor_radius)+'/'
		parameters = {'folder': folder_name, 'tumor_radius': str(tumor_radius)}
		create_xml_file('config/'+config_filename, folder_name+config_filename, parameters)

def run_model(tumor_radius_list, output_folder, config_filename):
	for tumor_radius in tumor_radius_list:
		print('Running tumor radius=' + str(tumor_radius))
		folder_name = output_folder+"output_TR"+str("%03d"%tumor_radius)+'/'

		# Write input for simulation & execute
		calling_model = ['./heterogeneity', folder_name+config_filename]
		cache = subprocess.run(calling_model, universal_newlines=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		if (cache.returncode != 0):
			print("Model output error! Returned: "+ str(cache.returncode))
			os._exit(1)

tumor_radius_list = [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]

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

create_config_folders(tumor_radius_list, output_folder, config_filename)
run_model(tumor_radius_list, output_folder, config_filename)