#pragma once


typedef struct
{
    unsigned int i,j,k;
} cell;

void sweep_octant(
        const unsigned int timestep, 
        const unsigned int oct, 
        const unsigned int ndiag, 
        const cell* cells,
        const int* num_cells,
        const unsigned int num_groups_todo);

void sweep_cell(
        const int istep,
        const int jstep,
        const int kstep,
        const unsigned int oct,
        const double* restrict l_flux_in,
        double* restrict l_flux_out,
        const cell * restrict cell_index,
        const unsigned int * restrict groups_todo,
        const unsigned int num_groups_todo,
        const unsigned int num_cells);

void compute_sweep_order(int** num_cells, cell** cells);

void perform_sweep(
        unsigned int num_groups_todo);


