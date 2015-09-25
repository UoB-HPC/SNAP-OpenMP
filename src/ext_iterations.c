#include "ext_sweep.h"
#include "ext_kernels.h"
#include "ext_shared.h"

void ext_reduce_angular_(void);

// Do the timestep, outer and inner iterations
void ext_iterations_(void)
{
	double *old_outer_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
	double *old_inner_scalar = malloc(sizeof(double)*nx*ny*nz*ng);
	double *new_scalar = malloc(sizeof(double)*nx*ny*nz*ng);

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

			calc_total_cross_section();
			calc_scattering_cross_section();
			calc_dd_coefficients();
			calc_time_delta();
			calc_denominator();

			// Compute the outer source
			calc_outer_source();

			// Save flux
			store_scalar_flux(old_outer_scalar);

			// Inner loop
			for (unsigned int i = 0; i < inners; i++)
			{
				tot_inners++;

				// Compute the inner source
				calc_inner_source();

				// Save flux
				store_scalar_flux(old_inner_scalar);
				zero_edge_flux_buffers();

#ifdef TIMING
				double t1 = omp_get_wtime();
#endif

				// Sweep
				perform_sweep(num_groups_todo);

#ifdef TIMING
				double t2 = omp_get_wtime();
				printf("sweep took: %lfs\n", t2-t1);
#endif

				// Scalar flux
				ext_reduce_angular_();
#ifdef TIMING
				double t3 = omp_get_wtime();
				printf("reductions took: %lfs\n", t3-t2);
#endif

				// Check convergence
				store_scalar_flux(new_scalar);

#ifdef TIMING
				double t4 = omp_get_wtime();
#endif

				inner_done = check_convergence(old_inner_scalar, new_scalar, epsi, groups_todo, &num_groups_todo, true);

#ifdef TIMING
				double t5 = omp_get_wtime();
				printf("inner conv test took %lfs\n",t5-t4);
#endif
				if (inner_done)
				{
					break;
				}
			}

			// Check convergence
			outer_done = check_convergence(old_outer_scalar, new_scalar, 100.0*epsi, groups_todo, &num_groups_todo, false);

			if (outer_done && inner_done)
			{
				break;
			}
		}

		printf("Time %d -  %d outers, %d inners.\n", t, tot_outers, tot_inners);

		// Exit the time loop early if outer not converged
		if (!outer_done)
		{
			break;
		}
	}

	double t2 = omp_get_wtime();

	printf("Time to convergence: %.3lfs\n", t2-t1);

	if (!outer_done)
	{
		printf("Warning: did not converge\n");
	}

	free(old_outer_scalar);
	free(new_scalar);
	free(old_inner_scalar);
	free(groups_todo);

    PRINT_PROFILING_RESULTS;
}

// Compute the scalar flux from the angular flux
void ext_reduce_angular_(void)
{
    START_PROFILING;

	double** angular = (global_timestep % 2 == 0) ? flux_out : flux_in;
	double** angular_prev = (global_timestep % 2 == 0) ? flux_in : flux_out;

#pragma omp parallel for
    for(int i = 0; i < nx; ++i)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int k = 0; k < nz; ++k)
            {
                for (unsigned int g = 0; g < ng; g++)
                {
                    for (unsigned int l = 0; l < cmom-1; l++)
                    {
                        scalar_mom(g,l,i,j,k) = 0.0;
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for(int i = 0; i < nx; ++i)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int k = 0; k < nz; ++k)
            {
                for (unsigned int g = 0; g < ng; g++)
                {
                    double tot_g = 0.0;

                    for (unsigned int a = 0; a < nang; a++)
                    {
                        // NOTICE: we do the reduction with psi, not ptr_out.
                        // This means that (line 307) the time dependant
                        // case isnt the value that is summed, but rather the
                        // flux in the cell
                        if (time_delta(g) != 0.0)
                        {
                            for(int o = 0; o < noct; ++o)
                            {
                                tot_g += weights(a) * (0.5 * (angular(o,a,g,i,j,k) + angular_prev(o,a,g,i,j,k)));
                            }

                            for (unsigned int l = 0; l < (cmom-1); l++)
                            {
                                for(int o = 0; o < noct; ++o)
                                {
                                    scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,o) * weights(a) * 
                                        (0.5 * (angular(o,a,g,i,j,k) + angular_prev(o,a,g,i,j,k)));
                                }
                            }
                        }
                        else
                        {
                            for(int o = 0; o < noct; ++o)
                            {
                                tot_g += weights(a) * angular(o,a,g,i,j,k);
                            }

                            for (unsigned int l = 0; l < (cmom-1); l++)
                            {
                                for(int o = 0; o < noct; ++o)
                                {
                                    scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,o) * weights(a) * angular(o,a,g,i,j,k);
                                }
                            }
                        }
                    }

                    scalar_flux(g,i,j,k) = tot_g;
                }
            }
        }
    }

    STOP_PROFILING(__func__);
}

