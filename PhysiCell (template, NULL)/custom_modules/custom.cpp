/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"

void create_cell_types( void )
{
	// set the random seed 
	SeedRandom( parameters.ints("random_seed") );  
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	/*cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.contact_function = contact_function;*/ 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}

void setup_tissue( void )
{
	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell 
	
	Cell* pC;
	
	for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	{
		Cell_Definition* pCD = cell_definitions_by_index[k]; 
		std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
		for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
		{
			std::vector<double> position = {0,0,0}; 
			position[0] = Xmin + UniformRandom()*Xrange; 
			position[1] = Ymin + UniformRandom()*Yrange; 
			position[2] = Zmin + UniformRandom()*Zrange; 
			
			pC = create_cell( *pCD ); 
			pC->assign_position( position );
		}
	}
	std::cout << std::endl; 
	
	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml(); 	
	
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{ return paint_by_number_cell_coloring(pCell); }

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 

void initialize_vectors(std::vector<int>& number_of_cells, std::vector<double>& tumor_volume) {
	int vector_size;

	if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::live_cells_cycle_model)
		vector_size = 2;
	else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::basic_Ki67_cycle_model)
		vector_size = 3;
	else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::advanced_Ki67_cycle_model
		|| cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_cycle_model)
		vector_size = 4;
	else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_separated_cycle_model)
		vector_size = 5;

	number_of_cells.resize(vector_size, 0);
	tumor_volume.resize(vector_size, 0);
}

std::string get_population_header(std::string sep) {
	std::string header = "times";
	if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::live_cells_cycle_model) {
		header += sep + "Live_num" + sep + "Dead_num";
		header += sep + "Live_vol" + sep + "Dead_vol";
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::basic_Ki67_cycle_model) {
		header += sep + "Ki67_negative_num" + sep + "Ki67_positive_num" + sep + "Dead_num";
		header += sep + "Ki67_negative_vol" + sep + "Ki67_positive_vol" + sep + "Dead_vol";
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::advanced_Ki67_cycle_model) {
		header += sep + "Ki67_negative_num" + sep + "Ki67_positive_premitotic_num" + sep + "Ki67_positive_postmitotic_num" + sep + "Dead_num";
		header += sep + "Ki67_negative_vol" + sep + "Ki67_positive_premitotic_vol" + sep + "Ki67_positive_postmitotic_vol" + sep + "Dead_vol";
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_cycle_model) {
		header += sep + "G0G1_phase_num" + sep + "S_phase_premitotic_num" + sep + "G2M_phase_postmitotic_num" + sep + "Dead_num";
		header += sep + "G0G1_phase_vol" + sep + "S_phase_premitotic_vol" + sep + "G2M_phase_postmitotic_vol" + sep + "Dead_vol";
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_separated_cycle_model) {
		header += sep + "G0G1_phase_num" + sep + "S_phase_premitotic_num" + sep + "G2_phase_postmitotic_num" + sep + "M_phase_postmitotic_num" + sep + "Dead_num";
		header += sep + "G0G1_phase_vol" + sep + "S_phase_premitotic_vol" + sep + "G2_phase_postmitotic_vol" + sep + "M_phase_postmitotic_vol" + sep + "Dead_vol";
	}

	return header;
}

std::string get_population_info(std::vector<int> number_of_cells, std::vector<double> tumor_volume, std::string sep) {
	std::string info = std::to_string(PhysiCell_globals.current_time);
	if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::live_cells_cycle_model) {
		info += sep + std::to_string(number_of_cells[0]) + sep + std::to_string(number_of_cells[1]);
		info += sep + std::to_string(tumor_volume[0]) + sep + std::to_string(tumor_volume[1]);
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::basic_Ki67_cycle_model) {
		info += sep + std::to_string(number_of_cells[0]) + sep + std::to_string(number_of_cells[1]) + sep + std::to_string(number_of_cells[2]);
		info += sep + std::to_string(tumor_volume[0]) + sep + std::to_string(tumor_volume[1]) + sep + std::to_string(tumor_volume[2]);
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::advanced_Ki67_cycle_model
		|| cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_cycle_model) {
		info += sep + std::to_string(number_of_cells[0]) + sep + std::to_string(number_of_cells[1]) + sep + std::to_string(number_of_cells[2]) + sep + std::to_string(number_of_cells[3]);
		info += sep + std::to_string(tumor_volume[0]) + sep + std::to_string(tumor_volume[1]) + sep + std::to_string(tumor_volume[2]) + sep + std::to_string(tumor_volume[3]);
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_separated_cycle_model) {
		info += sep + std::to_string(number_of_cells[0]) + sep + std::to_string(number_of_cells[1]) + sep + std::to_string(number_of_cells[2]) + sep + std::to_string(number_of_cells[3]) + sep + std::to_string(number_of_cells[4]);
		info += sep + std::to_string(tumor_volume[0]) + sep + std::to_string(tumor_volume[1]) + sep + std::to_string(tumor_volume[2]) + sep + std::to_string(tumor_volume[3]) + sep + std::to_string(tumor_volume[4]);
	}

	return info;
}

void count_cells(std::vector<int>& number_of_cells, std::vector<double>& tumor_volume) {
	int idx;

	if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::live_cells_cycle_model) {
		for (int i = 0; i < (*all_cells).size(); i++) {
			if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::live)
				idx = 0;
			else if ((*all_cells)[i]->phenotype.death.dead == true)
				idx = 1;

			number_of_cells[idx]++;
			tumor_volume[idx] += (*all_cells)[i]->phenotype.volume.total;
		}
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::basic_Ki67_cycle_model) {
		for (int i = 0; i < (*all_cells).size(); i++) {
			if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::Ki67_negative)
				idx = 0;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::Ki67_positive)
				idx = 1;
			else if ((*all_cells)[i]->phenotype.death.dead == true)
				idx = 2;

			number_of_cells[idx]++;
			tumor_volume[idx] += (*all_cells)[i]->phenotype.volume.total;
		}
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::advanced_Ki67_cycle_model) {
		for (int i = 0; i < (*all_cells).size(); i++) {
			if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::Ki67_negative)
				idx = 0;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::Ki67_positive_premitotic)
				idx = 1;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::Ki67_positive_postmitotic)
				idx = 2;
			else if ((*all_cells)[i]->phenotype.death.dead == true)
				idx = 3;

			number_of_cells[idx]++;
			tumor_volume[idx] += (*all_cells)[i]->phenotype.volume.total;
		}
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_cycle_model) {
		for (int i = 0; i < (*all_cells).size(); i++) {
			if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::G0G1_phase)
				idx = 0;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::S_phase)
				idx = 1;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::G2M_phase)
				idx = 2;
			else if ((*all_cells)[i]->phenotype.death.dead == true)
				idx = 3;

			number_of_cells[idx]++;
			tumor_volume[idx] += (*all_cells)[i]->phenotype.volume.total;
		}
	} else if (cell_defaults.phenotype.cycle.model().code == PhysiCell_constants::flow_cytometry_separated_cycle_model) {
		for (int i = 0; i < (*all_cells).size(); i++) {
			if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::G0G1_phase)
				idx = 0;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::S_phase)
				idx = 1;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::G2_phase)
				idx = 2;
			else if ((*all_cells)[i]->phenotype.cycle.current_phase().code == PhysiCell_constants::M_phase)
				idx = 3;
			else if ((*all_cells)[i]->phenotype.death.dead == true)
				idx = 4;

			number_of_cells[idx]++;
			tumor_volume[idx] += (*all_cells)[i]->phenotype.volume.total;
		}
	}
}