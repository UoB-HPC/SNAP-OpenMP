#include "ocl_sweep.h"

// Set the global timestep variable to the current timestep
void ocl_set_timestep_(const unsigned int *timestep)
{
	global_timestep = (*timestep) - 1;
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
	for(int i = 0; i < (cmom-1)*nx*ny*nz*ng; ++i)
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

void expand_scattering_cross_section(void)
{
	cl_int err;
	const size_t global[1] = {ng};

	err = clEnqueueNDRangeKernel(queue[0], k_calc_scattering_cross_section, 1, 0, global, NULL, 0, NULL, NULL);
}

bool check_convergence(double *old, double *new, double epsi, unsigned int *groups_todo, unsigned int *num_groups_todo, bool inner)
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

// Do the timestep, outer and inner iterations
void ocl_iterations_(void)
{
	cl_int err;
	double *old_outer_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
	double *old_inner_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
	double *new_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
	unsigned int *groups_todo = malloc(sizeof(unsigned int)*ng);
	unsigned int num_groups_todo;
	bool outer_done;
	double t1 = omp_get_wtime();

	// Timestep loop
	for (unsigned int t = 0; t < timesteps; t++)
	{
		unsigned int tot_outers = 0;
		unsigned int tot_inners = 0;
		global_timestep = t;

		// Calculate data required at the beginning of each timestep
		zero_scalar_flux();
		zero_flux_moments_buffer();

		// Outer loop
		outer_done = false;

		for (unsigned int o = 0; o < outers; o++)
		{
			// Reset the inner convergence list
			bool inner_done = false;
			for (unsigned int g = 0; g < ng; g++)
			{
				groups_todo[g] = g;
			}

			num_groups_todo = ng;
			tot_outers++;
			expand_cross_section(&xs, &total_cross_section);
			expand_scattering_cross_section();
			calc_dd_coefficients();
			calc_time_delta();
			calc_denom();

			// Compute the outer source
			calc_outer_source();

			// Save flux
			get_scalar_flux_(old_outer_scalar, false);

			// Inner loop
			for (unsigned int i = 0; i < inners; i++)
			{
				tot_inners++;
				// Compute the inner source
				calc_inner_source();
				// Save flux
				get_scalar_flux_(old_inner_scalar, false);
				zero_edge_flux_buffers_();
				// Copy over the list of groups to iterate on
				err = clEnqueueWriteBuffer(queue[0], groups_todo, CL_FALSE, 0, sizeof(unsigned int)*ng, groups_todo, 0, NULL, NULL);
				// Sweep
#ifdef TIMING
				double t1 = omp_get_wtime();
#endif
				ocl_sweep_(num_groups_todo);
#ifdef TIMING
				double t2 = omp_get_wtime();
				printf("sweep took: %lfs\n", t2-t1);
#endif
				// Scalar flux
				ocl_scalar_flux_();
#ifdef TIMING
				double t3 = omp_get_wtime();
				printf("reductions took: %lfs\n", t3-t2);
#endif
				// Check convergence
				get_scalar_flux_(new_scalar, true);
#ifdef TIMING
				double t4 = omp_get_wtime();
#endif
				inner_done = check_convergence(old_inner_scalar, new_scalar, epsi, groups_todo, &num_groups_todo, true);
#ifdef TIMING
				double t5 = omp_get_wtime();
				printf("inner conv test took %lfs\n",t5-t4);
#endif
				if (inner_done)
					break;
			}
			// Check convergence
			outer_done = check_convergence(old_outer_scalar, new_scalar, 100.0*epsi, groups_todo, &num_groups_todo, false);
			if (outer_done && inner_done)
				break;
		}
		printf("Time %d -  %d outers, %d inners.\n", t, tot_outers, tot_inners);
		// Exit the time loop early if outer not converged
		if (!outer_done)
			break;
	}
	double t2 = omp_get_wtime();

	printf("OpenCL: Time to convergence: %.3lfs\n", t2-t1);
	if (!outer_done)
		printf("Warning: did not converge\n");

	free(old_outer_scalar);
	free(new_scalar);
	free(old_inner_scalar);
	free(groups_todo);
}
