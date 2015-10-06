#pragma once

typedef struct 
{
    unsigned int num_cells;
    double* cells;

    // index is an index into the cells array for when storing the cell indexes
    unsigned int index;
} plane;

plane *compute_sweep_order(void);

void sweep_octant(
		const unsigned int timestep, 
		const unsigned int oct, 
		const unsigned int ndiag, 
		const plane *planes, 
		const unsigned int num_groups_todo);

void perform_sweep(
		unsigned int num_groups_todo);

void sweep_cell(
		const int istep,
		const int jstep,
		const int kstep,
		const unsigned int oct,
		const double* restrict l_flux_in,
		double* restrict l_flux_out,
		const double * restrict cell_index,
		const unsigned int num_groups_todo,
		const unsigned int num_cells);

