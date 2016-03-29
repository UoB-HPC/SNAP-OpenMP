#include <stdlib.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_kernels.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Compute the order of the sweep for the first octant
void compute_sweep_order()
{
    unsigned int nplanes = ichunk + ny + nz - 2;
    int* tmp_indices = (int*)malloc(nplanes*sizeof(int));

    for(int ii = 0; ii < nplanes; ++ii)
    {
        num_cells[ii] = 0;
        tmp_indices[ii] = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                num_cells[n]++;
            }
        }
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;

                unsigned int offset = 0;
                for(int l = 0; l < n; ++l)
                {
                    offset += num_cells[l];
                }

                unsigned int ind = tmp_indices[n];
                cells[offset + ind].i = i;
                cells[offset + ind].j = j;
                cells[offset + ind].k = k;
                tmp_indices[n]++;
            }
        }
    }

    free(tmp_indices);
}

// Sweep over the grid and compute the angular flux
void sweep_octant(
        const unsigned int timestep,
        const unsigned int oct,
        const unsigned int ndiag,
        const unsigned int num_groups_todo)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    //int order_3d = 0b000001010011100101110111;
    //int order_2d = 0b11100100;

    int order_3d = 342391;
    int order_2d = 228;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;

    size_t offset = oct*nang*nx*ny*nz*ng;
    double* l_flux_in = (timestep % 2 == 0 ? flux_in : flux_out) + offset;
    double* l_flux_out = (timestep % 2 == 0 ? flux_out : flux_in) + offset;

    int cells_processed = 0;
    for (unsigned int d = 0; d < ndiag; d++)
    {
        int ncells = num_cells[d];
        sweep_cell(istep, jstep, kstep, oct, l_flux_in, l_flux_out,
                &(cells[cells_processed]), groups_todo, num_groups_todo, ncells);
        cells_processed += ncells;
    }
}

// Perform a sweep over the grid for all the octants
void perform_sweep(
        unsigned int num_groups_todo)
{
    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    START_PROFILING;

    for (int o = 0; o < noct; o++)
    {
        sweep_octant(global_timestep, o, ndiag, num_groups_todo);
        zero_edge_flux_buffers();
    }

    STOP_PROFILING(__func__);
}

// Solve the transport equations for a single angle in a single cell for a single group
void sweep_cell(
        const int istep,
        const int jstep,
        const int kstep,
        const unsigned int oct,
        const double* restrict l_flux_in,
        double* restrict l_flux_out,
        const cell* restrict cell_index,
        const unsigned int * restrict groups_todo,
        const unsigned int num_groups_todo,
        const unsigned int ncells)
{
    double dd_i_temp = dd_i;
    double* restrict source_s = source;
    double* restrict mu_s = mu;
    double* restrict flux_i_s = flux_i;
    double* restrict flux_j_s = flux_j;
    double* restrict flux_k_s = flux_k;
    double* restrict scat_coeff_s = scat_coeff;
    double* restrict dd_j_s = dd_j;
    double* restrict dd_k_s = dd_k;
    double* restrict time_delta_s = time_delta;
    double* restrict denom_s = denom;
    double* restrict total_cross_section_s = total_cross_section;

#define scat_coeff_s(l,a,o) scat_coeff_s[l+(a*cmom)+(nang*cmom*(o))]
#define source_s(m,i,j,k,g) source_s[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define flux_i_s(a,g,j,k) flux_i_s[(a)+(nang*(g))+(nang*ng*(j))+(nang*ng*ny*(k))]
#define flux_j_s(a,g,i,k) flux_j_s[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(k))]
#define flux_k_s(a,g,i,j) flux_k_s[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))]
#define denom_s(a,g,i,j,k) denom_s[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define total_cross_section_s(g,i,j,k) total_cross_section_s[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define time_delta_s(g) time_delta_s[(g)]
#define mu_s(a) mu_s[(a)]
#define dd_j_s(a) dd_j_s[(a)]
#define dd_k_s(a) dd_k_s[(a)]

#pragma omp target teams distribute
    for(int nc = 0; nc < ncells; ++nc)
    {
        for(int tg = 0; tg < num_groups_todo; ++tg)
        {
            // Get indexes for angle and group
            const unsigned int i = (istep > 0) 
                ? cell_index[nc].i : nx - cell_index[nc].i - 1;
            const unsigned int j = (jstep > 0) 
                ? cell_index[nc].j : ny - cell_index[nc].j - 1;
            const unsigned int k = (kstep > 0) 
                ? cell_index[nc].k : nz - cell_index[nc].k - 1;
            const unsigned int g = groups_todo[tg];

#pragma omp simd
            for(int a = 0; a < nang; ++a)
            {
                // Assume transmissive (vacuum boundaries) and that we
                // are sweeping the whole grid so have access to all neighbours
                // This means that we only consider the case for one MPI task
                // at present.

                // Compute angular source
                // Begin with first scattering moment
                double source_term = source_s(0,i,j,k,g);

                // Add in the anisotropic scattering source moments
                for (unsigned int l = 1; l < cmom; l++)
                {
                    source_term += scat_coeff_s(l,a,oct) * source_s(l,i,j,k,g);
                }

                double psi = source_term
                    + flux_i_s(a,g,j,k)*mu_s(a)*dd_i_temp
                    + flux_j_s(a,g,i,k)*dd_j_s(a)
                    + flux_k_s(a,g,i,j)*dd_k_s(a);

                // Add contribution from last timestep flux if time-dependant
                if (time_delta_s(g) != 0.0)
                {
                    psi += time_delta_s(g) * l_flux_in(a,g,i,j,k);
                }

                psi *= denom_s(a,g,i,j,k);

                // Compute upwind fluxes
                double tmp_flux_i = 2.0*psi - flux_i_s(a,g,j,k);
                double tmp_flux_j = 2.0*psi - flux_j_s(a,g,i,k);
                double tmp_flux_k = 2.0*psi - flux_k_s(a,g,i,j);

                // Time differencing on final flux value
                if (time_delta_s(g) != 0.0)
                {
                    psi = 2.0 * psi - l_flux_in(a,g,i,j,k);
                }

                //int num_to_fix = 4;

                //// Fixup is a bounded loop as we will worst case fix up each face and centre value one after each other
                //double zeros[4];
                //for (int fix = 0; fix < 4; fix++)
                //{
                //    // Record which ones are zero
                //    zeros[0] = (tmp_flux_i < 0.0) ? 0.0 : 1.0;
                //    zeros[1] = (tmp_flux_j < 0.0) ? 0.0 : 1.0;
                //    zeros[2] = (tmp_flux_k < 0.0) ? 0.0 : 1.0;
                //    zeros[3] = (psi < 0.0) ? 0.0 : 1.0;

                //    if (num_to_fix == zeros[0] + zeros[1] + zeros[2] + zeros[3])
                //    {
                //        break;
                //    }

                //    num_to_fix = zeros[0] + zeros[1] + zeros[2] + zeros[3];

                //    // Recompute cell centre value
                //    psi = flux_i_s(a,g,j,k)*mu_s(a)*dd_i_temp*(1.0+zeros[0])
                //        + flux_j_s(a,g,j,k)*dd_j_s(a)*(1.0+zeros[1])
                //        + flux_k_s(a,g,i,j)*dd_k_s(a)*(1.0+zeros[2]);

                //    if (time_delta_s(g) != 0.0)
                //    {
                //        psi += time_delta_s(g) * l_flux_in(a,g,i,j,k) * (1.0+zeros[3]);
                //    }
                //    psi = 0.5*psi + source_term;

                //    double recalc_denom = total_cross_section_s(g,i,j,k);
                //    recalc_denom += mu_s(a) * dd_i_temp * zeros[0];
                //    recalc_denom += dd_j_s(a) * zeros[1];
                //    recalc_denom += dd_k_s(a) * zeros[2];
                //    recalc_denom += time_delta_s(g) * zeros[3];

                //    if (recalc_denom > 1.0E-12)
                //    {
                //        psi /= recalc_denom;
                //    }
                //    else
                //    {
                //        psi = 0.0;
                //    }

                //    // Recompute the edge fluxes with the new centre value
                //    tmp_flux_i = 2.0 * psi - flux_i_s(a,g,j,k);
                //    tmp_flux_j = 2.0 * psi - flux_j_s(a,g,i,k);
                //    tmp_flux_k = 2.0 * psi - flux_k_s(a,g,i,j);
                //    if (time_delta_s(g) != 0.0)
                //    {
                //        psi = 2.0*psi - l_flux_in(a,g,i,j,k);
                //    }
                //}

                //// Fix up loop is done, just need to set the final values
                //tmp_flux_i = tmp_flux_i * zeros[0];
                //tmp_flux_j = tmp_flux_j * zeros[1];
                //tmp_flux_k = tmp_flux_k * zeros[2];
                //psi = psi * zeros[3];

                // Write values to global memory
                flux_i_s(a,g,j,k) = tmp_flux_i;
                flux_j_s(a,g,i,k) = tmp_flux_j;
                flux_k_s(a,g,i,j) = tmp_flux_k;
                l_flux_out(a,g,i,j,k) = psi;
            }
        }
    }
}
