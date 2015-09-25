#include "ext_sweep.h"
#include "ext_shared.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
	for (unsigned int a_idx = 0; a_idx < nang; ++a_idx)
	{
		for (unsigned int g_idx = 0; g_idx < ng; ++g_idx)
		{
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
void calc_total_cross_section(void)
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

void calc_scattering_cross_section(void)
{
	for(unsigned int g = 0; g < ng; ++g)
	{
		for (unsigned int k = 0; k < nz; k++)
		{
			for (unsigned int j = 0; j < ny; j++)
			{
				for (unsigned int i = 0; i < nx; i++)
				{
					for (unsigned int l = 0; l < nmom; l++)
					{
						scat_cs(l,i,j,k,g) = gg_cs(map(i,j,k)-1,l,g,g);
					}
				}
			}
		}
	}
}

// Calculate the outer source
void calc_outer_source(void)
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

						g2g_source(0,i,j,k,g1) += gg_cs(map(i,j,k)-1,0,g2,g1) * scalar_flux(g2,i,j,k);

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
void calc_inner_source(void)
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
					source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar_flux(g,i,j,k);

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

void zero_edge_flux_buffers(void)
{
	for(int i = 0; i < nang*ny*nz*ng; ++i)
	{
		flux_i[i] = 0.0;
	}

	for(int i = 0; i < nang*nx*nz*ng; ++i)
	{
		flux_j[i] = 0.0;
	}

	for(int i = 0; i < nang*nx*ny*ng; ++i)
	{
		flux_k[i] = 0.0;
	}
}

void zero_flux_moments_buffer(void)
{
	for(int i = 0; i < cmom*nx*ny*nz*ng; ++i)
	{
		scalar_mom[i] = 0.0;
	}
}

void zero_scalar_flux(void)
{
	for(int i = 0; i < nx*ny*nz*ng; ++i)
	{
		scalar_flux[i] = 0.0;
	}
}

bool check_convergence(
		double *old, 
		double *new, 
		double epsi, 
		unsigned int *groups_todo, 
		unsigned int *num_groups_todo, 
		bool inner)
{
	bool r = true;

	// Reset the do_group list
	if (inner)
	{
		*num_groups_todo = 0;
	}

	for (unsigned int g = 0; g < ng; g++)
	{
		bool gr = false;
		for (unsigned int k = 0; k < nz; k++)
		{
			if (gr) break;
			for (unsigned int j = 0; j < ny; j++)
			{
				if (gr) break;
				for (unsigned int i = 0; i < nx; i++)
				{
					double val;
					if (fabs(old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] > tolr))
					{
						val = fabs(new[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)]/old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] - 1.0);
					}
					else
					{
						val = fabs(new[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)] - old[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)]);
					}

					if (val > epsi)
					{
						if (inner)
						{
							gr = true;
						}

						r = false;
						break;
					}
				}
			}
		}

		// Add g to the list of groups to do if we need to do it
		if (inner && gr)
		{
			groups_todo[*num_groups_todo] = g;
			*num_groups_todo += 1;
		}
	}

	// Check all inner groups are done in outer convergence test
	if (!inner)
	{
		if (*num_groups_todo != 0)
		{
			r = false;
		}
	}

	return r;
}

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
	for(int i = 0; i < nx*ny*nz*ng; ++i)
	{
		to[i] = scalar_flux[i];
	}
}
