#include <stdbool.h>
#include <math.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_problem.h"
#include "ext_profiler.h"
#include "ext_kernels.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
    START_PROFILING;

#pragma omp target teams distribute if(OFFLOAD)
//#pragma omp parallel for
    for (unsigned int ind = 0; ind < nx*ny*nz; ind++)
    {
        for (unsigned int g = 0; g < ng; ++g)
        {
            for (unsigned int a = 0; a < nang; ++a)
            {
                denom[a+g*nang+ind*ng*nang] = 1.0 / (total_cross_section[g+ind*ng] 
                        + time_delta(g) + mu(a)*dd_i + dd_j(a) + dd_k(a));
            }
        }
    }

    STOP_PROFILING(__func__);
}

// Calculate the time delta
void calc_time_delta(void)
{
    START_PROFILING;

#pragma omp target teams distribute if(OFFLOAD)
    for(int g = 0; g < ng; ++g)
    {
        time_delta(g) = 2.0 / (dt * velocity(g));
    }

    STOP_PROFILING(__func__);
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
    START_PROFILING;
        
    dd_i = 2.0 / dx;

//#pragma omp target update if(OFFLOAD) to(dd_i)

#pragma omp target teams distribute if(OFFLOAD)
    for(int a = 0; a < nang; ++a)
    {
        dd_j(a) = (2.0/dy)*eta(a);
        dd_k(a) = (2.0/dz)*xi(a);
    }

    STOP_PROFILING(__func__);
}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(void)
{
    START_PROFILING;

#pragma omp target teams distribute if(OFFLOAD)
    //#pragma omp parallel for
    for(int k = 0; k < nz; ++k)
    {
        for(int j = 0; j < ny; ++j)
        {
            for(int i = 0; i < nx; ++i)
            {
                for(int g = 0; g < ng; ++g)
                {
                    total_cross_section(g,i,j,k) = xs(mat(i,j,k)-1,g);
                }
            }
        }
    }

    STOP_PROFILING(__func__);
}

void calc_scattering_cross_section(void)
{
    START_PROFILING;

#pragma omp target teams distribute if(OFFLOAD)
    //#pragma omp parallel for
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
                        scat_cs(l,i,j,k,g) = gg_cs(mat(i,j,k)-1,l,g,g);
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__);
}

// Calculate the outer source
void calc_outer_source(void)
{
    START_PROFILING;

#pragma omp target teams distribute \
    collapse(4) if(OFFLOAD) 
    //#pragma omp parallel for 
    for (unsigned int g1 = 0; g1 < ng; g1++)
    {
        for(int k = 0; k < nz; ++k)
        {
            for(int j = 0; j < ny; ++j)
            {
                for(int i = 0; i < nx; ++i)
                {
                    g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);

                    for (unsigned int g2 = 0; g2 < ng; g2++)
                    {
                        if (g1 == g2)
                        {
                            continue;
                        }

                        g2g_source(0,i,j,k,g1) += gg_cs(mat(i,j,k)-1,0,g2,g1) 
                            * scalar_flux(g2,i,j,k);

                        unsigned int mom = 1;
                        for (unsigned int l = 1; l < nmom; l++)
                        {
                            for (int m = 0; m < lma(l); m++)
                            {
                                g2g_source(mom,i,j,k,g1) += gg_cs(mat(i,j,k)-1,l,g2,g1) 
                                    * scalar_mom(g2,mom-1,i,j,k);
                                mom++;
                            }
                        }
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__);
}

// Calculate the inner source
void calc_inner_source(void)
{
    START_PROFILING;

#pragma omp target teams distribute \
    collapse(4) if(OFFLOAD) 
    //#pragma omp parallel for
    for (unsigned int g = 0; g < ng; g++)
    {
        for(int k = 0; k < nz; ++k)
        {
            for(int j = 0; j < ny; ++j)
            {
                for(int i = 0; i < nx; ++i)
                {
                    source(0,i,j,k,g) = g2g_source(0,i,j,k,g) 
                        + scat_cs(0,i,j,k,g) * scalar_flux(g,i,j,k);

                    unsigned int mom = 1;
                    for (unsigned int l = 1; l < nmom; l++)
                    {
                        for (int m = 0; m < lma(l); m++)
                        {
                            source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) 
                                + scat_cs(l,i,j,k,g) * scalar_mom(g,mom-1,i,j,k);
                            mom++;
                        }
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__);
}

void zero_flux_in_out(void)
{
#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < flux_in_len; ++i)
    {
        flux_in[i] = 0.0;
    }

#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < flux_out_len; ++i)
    {
        flux_out[i] = 0.0;
    }
}

void zero_edge_flux_buffers(void)
{
#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < flux_i_len; ++i)
    {
        flux_i[i] = 0.0;
    }

#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < flux_j_len; ++i)
    {
        flux_j[i] = 0.0;
    }

#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < flux_k_len; ++i)
    {
        flux_k[i] = 0.0;
    }
}

void zero_flux_moments_buffer(void)
{
#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < scalar_mom_len; ++i)
    {
        scalar_mom[i] = 0.0;
    }
}

void zero_scalar_flux(void)
{
#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < scalar_flux_len; ++i)
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
    START_PROFILING;

    bool r = 1;

    int* temp_groups_todo = (int*)malloc(sizeof(int)*ng);

#pragma omp target teams distribute reduction(min: r)\
    map(from: temp_groups_todo[:ng]) if(OFFLOAD) 
    //#pragma omp parallel for
    for (unsigned int g = 0; g < ng; g++)
    {
        for (unsigned int ind = 0; ind < nx*ny*nz; ind++)
        {
            double val = (fabs(old[g+(ng*ind)] > tolr))
                ? fabs(new[g+(ng*ind)]/old[g+(ng*ind)] - 1.0)
                : fabs(new[g+(ng*ind)] - old[g+(ng*ind)]);

            if (val > epsi)
            {
                r = 0;

                if (inner)
                {
                    // Add g to the list of groups to do if we need to do it
                    temp_groups_todo[g] = 1;
                }

                break;
            }
        }
    }

    // Reorganise groups_todo
    for(int ii = 0; ii < ng; ++ii)
    {
        if(temp_groups_todo[ii])
        {
            groups_todo[*num_groups_todo] = ii;
            (*num_groups_todo)++;
        }
    }

    STOP_PROFILING(__func__);

    return r;
}

void initialise_device_memory(void)
{
    zero_scalar_flux();
    zero_flux_moments_buffer();
    zero_flux_in_out();
    zero_edge_flux_buffers();

#pragma omp target teams distribute if(OFFLOAD)
    //#pragma omp parallel for
    for(int ii = 0; ii < g2g_source_len; ++ii)
    {
        g2g_source[ii] = 0.0;
    }

#pragma omp target teams distribute if(OFFLOAD)
    //#pragma omp parallel for
    for(int ii = 0; ii < source_len; ++ii)
    {
        source[ii] = 0.0;
    }
}   

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
    START_PROFILING;

#pragma omp target teams distribute if(OFFLOAD) 
    //#pragma omp parallel for
    for(int i = 0; i < scalar_flux_len; ++i)
    {
        to[i] = scalar_flux[i];
    }

    STOP_PROFILING(__func__);
}
