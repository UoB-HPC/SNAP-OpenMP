#include "ext_sweep.h"
#include "ext_shared.h"

// Solve the transport equations for a single angle in a single cell for a single group
void sweep_cell(
		const int istep,
		const int jstep,
		const int kstep,
		const unsigned int oct,
		const struct cell * restrict cell_index,
		const unsigned int * restrict groups_todo,
		const unsigned int num_groups_todo,
		const unsigned int num_cells
		)
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

// Calculate the inverted denominator for all the energy groups
void calc_denominator(
		const unsigned int nx,
		const unsigned int ny,
		const unsigned int nz,
		const unsigned int nang,
		const unsigned int ng,
		const double * restrict total_cross_section,
		const double * restrict time_delta,
		const double * restrict mu,
		const double dd_i,
		const double * restrict dd_j,
		const double * restrict dd_k,
		double * restrict denom
		)
{
	const unsigned int a_idx = get_global_id(0);
	const unsigned int g_idx = get_global_id(1);

	for (unsigned int k = 0; k < nz; k++)
	{
		for (unsigned int j = 0; j < ny; j++)
		{
			for (unsigned int i = 0; i < nx; i++)
			{
				denom(a_idx,g_idx,i,j,k) = 1.0 / (total_cross_section(g_idx,i,j,k) 
						+ time_delta(g_idx) + mu(a_idx)*dd_i + dd_j(a_idx) + dd_k(a_idx));
			}
		}
	}
}

// Calculate the time delta
void calc_time_delta(void)
{
	for(int g = 0; g < ng; ++g)
	{
		time_delta(g) = 2.0 / (dt * velocity(g));
	}
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
	dd_i = 2.0 / dx;

	for(int a = 0; a < nang; ++a)
	{
		dd_j(a) = (2.0/dy)*eta(a);
		dd_k(a) = (2.0/dz)*xi(a);
	}
}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(
		const double * restrict xs,
		double * restrict total_cross_section
		)
{
	for(unsigned int g = 0; g < ng; ++g)
	{
		for (unsigned int k = 0; k < nz; k++)
		{
			for (unsigned int j = 0; j < ny; j++)
			{
				for (unsigned int i = 0; i < nx; i++)
				{
					total_cross_section(g,i,j,k) = xs(map(i,j,k)-1,g);
				}
			}
		}
	}
}

void calc_scattering_cross_section(
		const unsigned int nx,
		const unsigned int ny,
		const unsigned int nz,
		const unsigned int ng,
		const unsigned int nmom,
		const unsigned int nmat,
		const double * restrict gg_cs,
		const unsigned int * restrict map,
		double * restrict scat_cs
		)
{
	unsigned int g = get_global_id(0);

	for (unsigned int k = 0; k < nz; k++)
		for (unsigned int j = 0; j < ny; j++)
			for (unsigned int i = 0; i < nx; i++)
				for (unsigned int l = 0; l < nmom; l++)
					scat_cs(l,i,j,k,g) = gg_cs(map(i,j,k)-1,l,g,g);
}

// Calculate the outer source
void calc_outer_source()
{
	// Not sure if loop order is optimal...
	for(int i = 0; i < nx; ++i)
	{
		for(int j = 0; j < ny; ++j)
		{
			for(int k = 0; k < nz; ++k)
			{
				for (unsigned int g1 = 0; g1 < ng; g1++)
				{
					g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);

					for (unsigned int g2 = 0; g2 < ng; g2++)
					{
						if (g1 == g2)
						{
							continue;
						}

						g2g_source(0,i,j,k,g1) += gg_cs(map(i,j,k)-1,0,g2,g1) * scalar(g2,i,j,k);

						unsigned int mom = 1;
						for (unsigned int l = 1; l < nmom; l++)
						{
							for (int m = 0; m < lma(l); m++)
							{
								g2g_source(mom,i,j,k,g1) += gg_cs(map(i,j,k)-1,l,g2,g1) * scalar_mom(g2,mom-1,i,j,k);
								mom++;
							}
						}
					}
				}
			}
		}
	}
}

// Calculate the inner source
void calc_inner_source()
{
	// Not sure if loop order is optimal...
	for(int i = 0; i < nx; ++i)
	{
		for(int j = 0; j < ny; ++j)
		{
			for(int k = 0; k < nz; ++k)
			{
				for (unsigned int g = 0; g < ng; g++)
				{
					source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar(g,i,j,k);

					unsigned int mom = 1;
					for (unsigned int l = 1; l < nmom; l++)
					{
						for (int m = 0; m < lma(l); m++)
						{
							source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) + scat_cs(l,i,j,k,g) * scalar_mom(g,mom-1,i,j,k);
							mom++;
						}
					}
				}
			}
		}
	}
}
