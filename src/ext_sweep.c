#include "ext_sweep.h"

// Compute the order of the sweep for the first octant
plane *compute_sweep_order(void)
{
    unsigned int nplanes = ichunk + ny + nz - 2;
    plane *planes = (plane *)malloc(sizeof(plane)*nplanes);
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].num_cells = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                planes[n].num_cells++;
            }
        }
    }

    // Allocate the memory for each plane
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].cells = (struct cell *)malloc(sizeof(struct cell)*planes[i].num_cells);
        planes[i].index = 0;
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                unsigned int idx = planes[n].index;
                planes[n].cells[idx].i = i;
                planes[n].cells[idx].j = j;
                planes[n].cells[idx].k = k;
                planes[n].index += 1;
            }
        }
    }

    return planes;
}

// Sweep over the grid and compute the angular flux
void sweep_octant(
		const unsigned int timestep, 
		const unsigned int oct, 
		const unsigned int ndiag, 
		const plane *planes, 
		const unsigned int num_groups_todo)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    int order_3d = 0b000001010011100101110111;

    int order_2d = 0b11100100;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;

	for (unsigned int d = 0; d < ndiag; d++)
	{
		sweep_cell(istep, jstep, kstep, oct, cell_index, groups_todo, num_groups_todo, planes[d].num_cells);
	}
}

// Perform a sweep over the grid for all the octants
void perform_sweep(
		unsigned int num_groups_todo)
{
	// Number of planes in this octant
	unsigned int ndiag = ichunk + ny + nz - 2;

	// Get the order of cells to enqueue
	plane *planes = compute_sweep_order();

	for (int o = 0; o < noct; o++)
	{
		enqueue_octant(global_timestep, o, ndiag, planes, num_groups_todo);
		zero_edge_flux_buffers();
	}

	// Free planes
	for (unsigned int i = 0; i < ndiag; i++)
	{
		free(planes[i].cells);
	}

	free(planes);
}

// Solve the transport equations for a single angle in a single cell for a single group
void sweep_cell(
		const int istep,
		const int jstep,
		const int kstep,
		const unsigned int oct,
		const struct cell * restrict cell_index,
		const unsigned int * restrict groups_todo,
		const unsigned int num_groups_todo,
		const unsigned int num_cells)
{
	for(int a_idx = 0; a_idx < nang; ++a_idx)
	{
		for(int tmp_g_idx = 0; tmp_g_idx < num_groups_todo; ++tmp_g_idx)
		{
			for(int nc = 0; nc < num_cells; ++nc)
			{
				// Get indexes for angle and group
				const unsigned int i = (istep > 0) ? cell_index[nc].i : nx - cell_index[nc].i - 1;
				const unsigned int j = (jstep > 0) ? cell_index[nc].j : ny - cell_index[nc].j - 1;
				const unsigned int k = (kstep > 0) ? cell_index[nc].k : nz - cell_index[nc].k - 1;
				const unsigned int g_idx = groups_todo[tmp_g_idx];

				// Assume transmissive (vacuum boundaries) and that we
				// are sweeping the whole grid so have access to all neighbours
				// This means that we only consider the case for one MPI task
				// at present.

				// Compute angular source
				// Begin with first scattering moment
				double source_term = source(0,i,j,k,g_idx);

				// Add in the anisotropic scattering source moments
				for (unsigned int l = 1; l < cmom; l++)
				{
					source_term += scat_coef(a_idx,l,oct) * source(l,i,j,k,g_idx);
				}

				double psi = source_term 
					+ flux_i(a_idx,g_idx,j,k)*mu(a_idx)*dd_i 
					+ flux_j(a_idx,g_idx,i,k)*dd_j(a_idx) 
					+ flux_k(a_idx,g_idx,i,j)*dd_k(a_idx);

				// Add contribution from last timestep flux if time-dependant
				if (time_delta(g_idx) != 0.0)
				{
					psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k);
				}

				psi *= denom(a_idx,g_idx,i,j,k);

				// Compute upwind fluxes
				double tmp_flux_i = 2.0*psi - flux_i(a_idx,g_idx,j,k);
				double tmp_flux_j = 2.0*psi - flux_j(a_idx,g_idx,i,k);
				double tmp_flux_k = 2.0*psi - flux_k(a_idx,g_idx,i,j);

				// Time differencing on final flux value
				if (time_delta(g_idx) != 0.0)
				{
					psi = 2.0 * psi - flux_in(a_idx,g_idx,i,j,k);
				}

				// Perform the fixup loop
				double zeros[4] = {1.0, 1.0, 1.0, 1.0};
				int num_to_fix = 4;
				// Fixup is a bounded loop as we will worst case fix up each face and centre value one after each other
				for (int fix = 0; fix < 4; fix++)
				{
					// Record which ones are zero
					if (tmp_flux_i < 0.0) zeros[0] = 0.0;
					if (tmp_flux_j < 0.0) zeros[1] = 0.0;
					if (tmp_flux_k < 0.0) zeros[2] = 0.0;
					if (psi < 0.0) zeros[3] = 0.0;

					if (num_to_fix == zeros[0] + zeros[1] + zeros[2] + zeros[3])
					{
						// We have fixed up enough
						break;
					}
					num_to_fix = zeros[0] + zeros[1] + zeros[2] + zeros[3];

					// Recompute cell centre value
					psi = flux_i(a_idx,g_idx,j,k)*mu(a_idx)*dd_i*(1.0+zeros[0]) 
						+ flux_j(a_idx,g_idx,j,k)*dd_j(a_idx)*(1.0+zeros[1]) 
						+ flux_k(a_idx,g_idx,i,j)*dd_k(a_idx)*(1.0+zeros[2]);

					if (time_delta(g_idx) != 0.0)
					{
						psi += time_delta(g_idx) * flux_in(a_idx,g_idx,i,j,k) * (1.0+zeros[3]);
					}
					psi = 0.5*psi + source_term;
					double recalc_denom = total_cross_section(g_idx,i,j,k);
					recalc_denom += mu(a_idx) * dd_i * zeros[0];
					recalc_denom += dd_j(a_idx) * zeros[1];
					recalc_denom += dd_k(a_idx) * zeros[2];
					recalc_denom += time_delta(g_idx) * zeros[3];

					if (recalc_denom > 1.0E-12)
					{
						psi /= recalc_denom;
					}
					else
					{
						psi = 0.0;
					}

					// Recompute the edge fluxes with the new centre value
					tmp_flux_i = 2.0 * psi - flux_i(a_idx,g_idx,j,k);
					tmp_flux_j = 2.0 * psi - flux_j(a_idx,g_idx,i,k);
					tmp_flux_k = 2.0 * psi - flux_k(a_idx,g_idx,i,j);
					if (time_delta(g_idx) != 0.0)
					{
						psi = 2.0*psi - flux_in(a_idx,g_idx,i,j,k);
					}
				}
				// Fix up loop is done, just need to set the final values
				tmp_flux_i = tmp_flux_i * zeros[0];
				tmp_flux_j = tmp_flux_j * zeros[1];
				tmp_flux_k = tmp_flux_k * zeros[2];
				psi = psi * zeros[3];

				// Write values to global memory
				flux_i(a_idx,g_idx,j,k) = tmp_flux_i;
				flux_j(a_idx,g_idx,i,k) = tmp_flux_j;
				flux_k(a_idx,g_idx,i,j) = tmp_flux_k;
				flux_out(a_idx,g_idx,i,j,k) = psi;
			}
		}
	}
}
