#include <stdbool.h>
#include <math.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: total_cross_section[:total_cross_section_len], time_delta[:time_delta_len], \
            mu[:mu_len], denom[:denom_len], dd_j[:dd_j_len], dd_k[:dd_k_len])
#pragma omp parallel for
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3)
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

    STOP_PROFILING(__func__, true);
}

// Calculate the time delta
void calc_time_delta(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: velocity[:velocity_len], time_delta[:time_delta_len])
#pragma omp parallel for
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3)
    for(int g = 0; g < ng; ++g)
    {
        time_delta(g) = 2.0 / (dt * velocity(g));
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: eta[:eta_len], xi[:xi_len], dd_j[:dd_j_len], dd_k[:dd_k_len])
    {
        dd_i = 2.0 / dx;

#pragma omp parallel for
        //#pragma omp target teams distribute parallel for \
        //num_teams(59) num_threads(3)
        for(int a = 0; a < nang; ++a)
        {
            dd_j(a) = (2.0/dy)*eta(a);
            dd_k(a) = (2.0/dz)*xi(a);
        }
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: xs[:xs_len], mat[:mat_len], total_cross_section[:total_cross_section_len])
#pragma omp parallel for
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3)
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

    STOP_PROFILING(__func__, true);
}

void calc_scattering_cross_section(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: gg_cs[:gg_cs_len], mat[:mat_len], scat_cs[:scat_cs_len])
#pragma omp parallel for
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3)
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

    STOP_PROFILING(__func__, true);
}

// Calculate the outer source
void calc_outer_source(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: fixed_source[:fixed_source_len], gg_cs[:gg_cs_len], mat[:mat_len], lma[:lma_len], \
            g2g_source[:g2g_source_len], scalar_mom[:scalar_mom_len], scalar_flux[:scalar_flux_len])
#pragma omp parallel for collapse(4)
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3) collapse(4)
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

                        g2g_source(0,i,j,k,g1) += gg_cs(mat(i,j,k)-1,0,g2,g1) * scalar_flux(g2,i,j,k);

                        unsigned int mom = 1;
                        for (unsigned int l = 1; l < nmom; l++)
                        {
                            for (int m = 0; m < lma(l); m++)
                            {
                                g2g_source(mom,i,j,k,g1) += gg_cs(mat(i,j,k)-1,l,g2,g1) * scalar_mom(g2,mom-1,i,j,k);
                                mom++;
                            }
                        }
                    }
                }
            }
        }
    }

    STOP_PROFILING(__func__, true);
}

// Calculate the inner source
void calc_inner_source(void)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: scat_cs[:scat_cs_len], source[:source_len], \
            g2g_source[:g2g_source_len], scalar_flux[:scalar_flux_len], \
            scalar_mom[:scalar_mom_len], lma[:lma_len])
#pragma omp parallel for
    //#pragma omp target teams distribute parallel for \
    //num_teams(59) num_threads(3) collapse(4)
    for (unsigned int g = 0; g < ng; g++)
    {
        for(int k = 0; k < nz; ++k)
        {
            for(int j = 0; j < ny; ++j)
            {
                for(int i = 0; i < nx; ++i)
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

    STOP_PROFILING(__func__, true);
}

void zero_edge_flux_buffers(void)
{
#pragma omp target if(OFFLOAD) \
    map(alloc: flux_i[:flux_i_len])
#pragma omp parallel for
    for(int i = 0; i < flux_i_len; ++i)
    {
        flux_i[i] = 0.0;
    }

#pragma omp target if(OFFLOAD) \
    map(alloc: flux_j[:flux_j_len])
#pragma omp parallel for
    for(int i = 0; i < flux_j_len; ++i)
    {
        flux_j[i] = 0.0;
    }

#pragma omp target if(OFFLOAD) \
    map(alloc: flux_k[:flux_k_len])
#pragma omp parallel for
    for(int i = 0; i < flux_k_len; ++i)
    {
        flux_k[i] = 0.0;
    }
}

void zero_flux_moments_buffer(void)
{
#pragma omp target if(OFFLOAD) \
    map(alloc: scalar_mom[:scalar_mom_len])
#pragma omp parallel for
    //#pragma omp target if(OFFLOAD) teams distribute parallel for \
    //num_teams(59) num_threads(3)
    for(int i = 0; i < scalar_mom_len; ++i)
    {
        scalar_mom[i] = 0.0;
    }
}

void zero_flux_in_out(void)
{
#pragma omp target if(OFFLOAD) \
    map(alloc: flux_in[:flux_in_len], flux_out[:flux_out_len])
#pragma omp parallel for
    //#pragma omp target if(OFFLOAD) teams distribute parallel for \
    //num_teams(59) num_threads(3)
    for(int i = 0; i < flux_in_len; ++i)
    {
        flux_in[i] = 0.0;
        flux_out[i] = 0.0;
    }
}

void zero_scalar_flux(void)
{
#pragma omp target if(OFFLOAD) \
    map(alloc: scalar_flux[:scalar_flux_len])
#pragma omp parallel for
    //#pragma omp target if(OFFLOAD) teams distribute parallel for \
    //num_teams(59) num_threads(3)
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

    bool r = true;

    // Reset the do_group list
    if (inner)
    {
        *num_groups_todo = 0;
    }

#pragma omp target if(OFFLOAD) \
    map(alloc: old[:scalar_flux_len], new[:scalar_flux_len], groups_todo[:groups_todo_len])
#pragma omp parallel for
    //#pragma omp target if(OFFLOAD) teams distribute parallel for \
    //num_teams(59) num_threads(3)
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

    STOP_PROFILING(__func__, true);

    return r;
}

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
    START_PROFILING;

#pragma omp target if(OFFLOAD) \
    map(alloc: scalar_flux[:scalar_flux_len], to[:scalar_flux_len])
#pragma omp parallel for
    //#pragma omp target if(OFFLOAD) teams distribute parallel for \
    //num_teams(59) num_threads(3)
    for(int i = 0; i < scalar_flux_len; ++i)
    {
        to[i] = scalar_flux[i];
    }

    STOP_PROFILING(__func__, true);
}
