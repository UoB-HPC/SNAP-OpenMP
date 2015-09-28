#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "ext_core.h"
#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_kernels.h"
#include "ext_problem.h"
#include "ext_profiler.h"

#define MIC_DEVICE 0

void ext_solve_(
		double *mu, 
		double *eta, 
		double *xi,
		double *scat_coeff,
		double *weights,
		double *velocity,
		double *xs,
		int *mat,
		double *fixed_source,
		double *gg_cs,
		int *lma)
{
    initialise_host_memory();

#pragma omp target update to(nx, ny, nz, ng, nang, noct, cmom, nmom, \
        nmat, ichunk, timesteps, dt, dx, dy, dz, outers, inners, \
        epsi, tolr, dd_i, global_timestep)

#pragma omp target device(MIC_DEVICE) if(OFFLOAD) \
        map(to: mu[:nang], eta[:nang], xi[:nang], \
            scat_coeff[:nang*nmom*noct], weights[:nang], mat[:nx*ny*nz], \
            velocity[:ng], xs[:nx*ng], fixed_source[:nx*ny*nz*ng], \
            gg_cs[:nmom*nmom*ng*ng], lma[:nmom]) \
        map(from: scalar_flux[:nx*ny*nz*ng], flux_in[:nang*nx*ny*nz*ng*noct],\
            flux_out[:nang*nx*ny*nz*ng*noct], scalar_mom[:cmom*nx*ny*nz*ng])
    {
        initialise_device_memory(mu, eta, xi, scat_coeff, weights, velocity,
                xs, mat, fixed_source, gg_cs, lma);

        iterate();
    }
}

// Initialises the problem parameters
void ext_initialise_parameters_(
        int *nx_, int *ny_, int *nz_,
        int *ng_, int *nang_, int *noct_,
        int *cmom_, int *nmom_,
        int *ichunk_,
        double *dx_, double *dy_, double *dz_,
        double *dt_,
        int *nmat_,
        int *timesteps_, int *outers_, int *inners_,
        double *epsi_, double *tolr_)
{
    START_PROFILING;

    // Save problem size information to globals
    nx = *nx_;
    ny = *ny_;
    nz = *nz_;
    ng = *ng_;
    nang = *nang_;
    noct = *noct_;
    cmom = *cmom_;
    nmom = *nmom_;
    ichunk = *ichunk_;
    dx = *dx_;
    dy = *dy_;
    dz = *dz_;
    dt = *dt_;
    nmat = *nmat_;
    timesteps = *timesteps_;
    outers = *outers_;
    inners = *inners_;

    epsi = *epsi_;
    tolr = *tolr_;

    if (nx != ichunk)
    {
        printf("Warning: nx and ichunk are different - expect the answers to be wrong...\n");
    }

    STOP_PROFILING(__func__);
}

// Argument list:
// nx, ny, nz are the (local to MPI task) dimensions of the grid
// ng is the number of energy groups
// cmom is the "computational number of moments"
// ichunk is the number of yz planes in the KBA decomposition
// dd_i, dd_j(nang), dd_k(nang) is the x,y,z (resp) diamond difference coefficients
// mu(nang) is x-direction cosines
// scat_coef [ec](nang,cmom,noct) - Scattering expansion coefficients
// time_delta [vdelt](ng)              - time-absorption coefficient
// denom(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
// weights(nang) - angle weights for scalar reduction
void initialise_device_memory(
        double *mu_in, 
        double *eta_in, 
        double *xi_in,
        double *scat_coeff_in,
        double *weights_in,
        double *velocity_in,
        double *xs_in,
        int *mat_in,
        double *fixed_source_in,
        double *gg_cs_in,
        int *lma_in)
{
    START_PROFILING;

    // Allocate flux in and out

    for(int i = 0; i < nang*nx*ny*nz*ng*noct; ++i)
    {
        flux_in[i] = 0.0;
        flux_out[i] = 0.0;
    }

    // flux_i(nang,ny,nz,ng)     - Working psi_x array (edge pointers)
    // flux_j(nang,ichunk,nz,ng) - Working psi_y array
    // flux_k(nang,ichunk,ny,ng) - Working psi_z array

    flux_i = malloc(sizeof(double)*nang*ny*nz*ng);
    flux_j = malloc(sizeof(double)*nang*nx*nz*ng);
    flux_k = malloc(sizeof(double)*nang*nx*ny*ng);
    zero_edge_flux_buffers();

    zero_flux_moments_buffer();

    zero_scalar_flux();

    dd_j = malloc(sizeof(double)*nang);
    dd_k = malloc(sizeof(double)*nang);
    total_cross_section = malloc(sizeof(double)*nx*ny*nz*ng);
    scat_cs = malloc(sizeof(double)*nmom*nx*ny*nz*ng);
    denom = malloc(sizeof(double)*nang*nx*ny*nz*ng);
    source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
    time_delta = malloc(sizeof(double)*ng);
    groups_todo = malloc(sizeof(unsigned int)*ng);
    g2g_source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);

    // Read-only buffers initialised in Fortran code
    mu = mu_in;
    eta = eta_in;
    xi = xi_in;
    scat_coeff = scat_coeff_in;
    weights = weights_in;
    velocity = velocity_in;
    mat = mat_in;
    fixed_source = fixed_source_in;
    gg_cs = gg_cs_in;
    lma = lma_in;
    xs = xs_in;

    STOP_PROFILING(__func__);
}

// Initialises buffers required on the host
void initialise_host_memory(void)
{
    scalar_flux = malloc(sizeof(double)*nx*ny*nz*ng);
    flux_in = malloc(sizeof(double)*nang*nx*ny*nz*ng*noct);
    flux_out = malloc(sizeof(double)*nang*nx*ny*nz*ng*noct);
    scalar_mom = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
}

// Do the timestep, outer and inner iterations
void iterate(void)
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
                reduce_angular();
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
void reduce_angular(void)
{
    START_PROFILING;

    double* angular = (global_timestep % 2 == 0) ? flux_out : flux_in;
    double* angular_prev = (global_timestep % 2 == 0) ? flux_in : flux_out;

#pragma omp parallel for collapse(3)
    for(int k = 0; k < nz; ++k)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int i = 0; i < nx; ++i)
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
                                tot_g += weights(a) * (0.5 * (angular(0,a,g,i,j,k) + angular_prev(0,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(1,a,g,i,j,k) + angular_prev(1,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(2,a,g,i,j,k) + angular_prev(2,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(3,a,g,i,j,k) + angular_prev(3,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(4,a,g,i,j,k) + angular_prev(4,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(5,a,g,i,j,k) + angular_prev(5,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(6,a,g,i,j,k) + angular_prev(6,a,g,i,j,k)));
                                tot_g += weights(a) * (0.5 * (angular(7,a,g,i,j,k) + angular_prev(7,a,g,i,j,k)));

                            for (unsigned int l = 0; l < (cmom-1); l++)
                            {
                                scalar_mom(g,l,i,j,k) = scat_coeff(a,l+1,0) * weights(a) * (0.5 * (angular(0,a,g,i,j,k) + angular_prev(0,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,1) * weights(a) * (0.5 * (angular(1,a,g,i,j,k) + angular_prev(1,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,2) * weights(a) * (0.5 * (angular(2,a,g,i,j,k) + angular_prev(2,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,3) * weights(a) * (0.5 * (angular(3,a,g,i,j,k) + angular_prev(3,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,4) * weights(a) * (0.5 * (angular(4,a,g,i,j,k) + angular_prev(4,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,5) * weights(a) * (0.5 * (angular(5,a,g,i,j,k) + angular_prev(5,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,6) * weights(a) * (0.5 * (angular(6,a,g,i,j,k) + angular_prev(6,a,g,i,j,k)));
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,7) * weights(a) * (0.5 * (angular(7,a,g,i,j,k) + angular_prev(7,a,g,i,j,k)));
                            }
                        }
                        else
                        {
                            tot_g += weights(a) * angular(0,a,g,i,j,k);
                            tot_g += weights(a) * angular(1,a,g,i,j,k);
                            tot_g += weights(a) * angular(2,a,g,i,j,k);
                            tot_g += weights(a) * angular(3,a,g,i,j,k);
                            tot_g += weights(a) * angular(4,a,g,i,j,k);
                            tot_g += weights(a) * angular(5,a,g,i,j,k);
                            tot_g += weights(a) * angular(6,a,g,i,j,k);
                            tot_g += weights(a) * angular(7,a,g,i,j,k);

                            for (unsigned int l = 0; l < (cmom-1); l++)
                            {
                                scalar_mom(g,l,i,j,k) = scat_coeff(a,l+1,0) * weights(a) * angular(0,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,1) * weights(a) * angular(1,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,2) * weights(a) * angular(2,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,3) * weights(a) * angular(3,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,4) * weights(a) * angular(4,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,5) * weights(a) * angular(5,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,6) * weights(a) * angular(6,a,g,i,j,k);
                                scalar_mom(g,l,i,j,k) += scat_coeff(a,l+1,7) * weights(a) * angular(7,a,g,i,j,k);
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

// Copy the scalar flux value back to the host and transpose
void ext_get_transpose_scalar_flux_(double *scalar)
{
    //START_PROFILING;

    // Transpose the data into the original SNAP format
    for (unsigned int g = 0; g < ng; g++)
    {
        for (unsigned int k = 0; k < nz; k++)
        {
            for (unsigned int j = 0; j < ny; j++)
            {
                for (unsigned int i = 0; i < nx; i++)
                {
                    scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] 
                        = scalar_flux[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)];
                }
            }
        }
    }

    //STOP_PROFILING(__func__);
}

void ext_get_transpose_scalar_moments_(double *scalar_moments)
{
    //START_PROFILING;

    // Transpose the data into the original SNAP format
    for (unsigned int g = 0; g < ng; g++)
    {
        for (unsigned int l = 0; l < cmom-1; l++)
        {
            for (unsigned int k = 0; k < nz; k++)
            {
                for (unsigned int j = 0; j < ny; j++)
                {
                    for (unsigned int i = 0; i < nx; i++)
                    {
                        scalar_moments[l+((cmom-1)*i)+((cmom-1)*nx*j)+((cmom-1)*nx*ny*k)+((cmom-1)*nx*ny*nz*g)] 
                            = scalar_mom[g+(ng*l)+(ng*(cmom-1)*i)+(ng*(cmom-1)*nx*j)+(ng*(cmom-1)*nx*ny*k)];
                    }
                }
            }
        }
    }

    //STOP_PROFILING(__func__);
}

// Copy the flux_out buffer back to the host
void ext_get_transpose_output_flux_(double* output_flux)
{
    //START_PROFILING;

    double *tmp = (global_timestep % 2 == 0) ? flux_out : flux_in;

    // Transpose the data into the original SNAP format
    for (int a = 0; a < nang; a++)
    {
        for (int g = 0; g < ng; g++)
        {
            for (int k = 0; k < nz; k++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        for (int o = 0; o < noct; o++)
                        {
                            output_flux[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)] 
                                = tmp[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*nx*ny*nz*ng*o)];
                        }
                    }
                }
            }
        }
    }

    //STOP_PROFILING(__func__);
}
