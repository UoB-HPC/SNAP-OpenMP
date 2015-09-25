#include "ext_problem.h"
#include "ext_kernels.h"

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
void ext_initialise_memory_(
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
		int *lma_in,
		double *g2g_source_in)
{
	// Create zero array for the edge flux buffers
	// First we need maximum two of nx, ny and nz
	size_t s = nang * ng;
	if (nx < ny && nx < nz)
	{
		s *= ny * nz;
	}
	else if (ny < nx && ny < nz)
	{
		s *= nx * nz;
	}
	else
	{
		s *= nx * ny;
	}

	flux_in = (double**)malloc(sizeof(double*)*noct);
	flux_out = (double**)malloc(sizeof(double*)*noct);

	// Allocate flux in and out
	for (unsigned int o = 0; o < noct; o++)
	{
		flux_in[o] = (double*)malloc(sizeof(double)*nang*nx*ny*nz*ng);
		flux_out[o] = (double*)malloc(sizeof(double)*nang*nx*ny*nz*ng);

		// Zero centre
		for(int i = 0; i < nang*nx*ny*nz*ng; ++i)
		{
			flux_in[o][i] = 0.0;
			flux_out[o][i] = 0.0;
		}
	}

	// flux_i(nang,ny,nz,ng)     - Working psi_x array (edge pointers)
	// flux_j(nang,ichunk,nz,ng) - Working psi_y array
	// flux_k(nang,ichunk,ny,ng) - Working psi_z array

	flux_i = malloc(sizeof(double)*nang*ny*nz*ng);
	flux_j = malloc(sizeof(double)*nang*nx*nz*ng);
	flux_k = malloc(sizeof(double)*nang*nx*ny*ng);
	zero_edge_flux_buffers();

	scalar_mom = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
	zero_flux_moments_buffer();

	scalar_flux = malloc(sizeof(double)*nx*ny*nz*ng);
	zero_scalar_flux();

	dd_j = malloc(sizeof(double)*nang);
	dd_k = malloc(sizeof(double)*nang);
	total_cross_section = malloc(sizeof(double)*nx*ny*nz*ng);
	scat_cs = malloc(sizeof(double)*nmom*nx*ny*nz*ng);
	denom = malloc(sizeof(double)*nang*nx*ny*nz*ng);
	source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
	time_delta = malloc(sizeof(double)*ng);
	groups_todo = malloc(sizeof(unsigned int)*ng);

	// Read-only buffers initialised in Fortran code
	mu = mu_in;
	eta = eta_in;
	xi = xi_in;
	scat_coeff = scat_coeff_in;
	weights = weights_in;
	velocity = velocity_in;
	map = mat_in;
	fixed_source = fixed_source_in;
	gg_cs = gg_cs_in;
	lma = lma_in;

	// Buffers copied in from Fortran
	xs = malloc(sizeof(double)*nmat*ng);
	memcpy(xs, xs_in, sizeof(double)*nmat*ng);

	g2g_source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
	memcpy(g2g_source, g2g_source_in, sizeof(double)*cmom*nx*ny*nz*ng);
}

// Copy the scalar flux value back to the host and transpose
void ext_get_transpose_scalar_flux_(double *scalar)
{
	// Transpose the data into the original SNAP format
	for (unsigned int g = 0; g < ng; g++)
	{
		for (unsigned int i = 0; i < nx; i++)
		{
			for (unsigned int j = 0; j < ny; j++)
			{
				for (unsigned int k = 0; k < nz; k++)
				{
					scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] 
						= scalar_flux[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)];
				}
			}
		}
	}
}

void ext_get_transpose_scalar_moments_(double *scalar_moments)
{
	// Transpose the data into the original SNAP format
	for (unsigned int g = 0; g < ng; g++)
	{
		for (unsigned int l = 0; l < cmom-1; l++)
		{
			for (unsigned int i = 0; i < nx; i++)
			{
				for (unsigned int j = 0; j < ny; j++)
				{
					for (unsigned int k = 0; k < nz; k++)
					{
						scalar_moments[l+((cmom-1)*i)+((cmom-1)*nx*j)+((cmom-1)*nx*ny*k)+((cmom-1)*nx*ny*nz*g)] 
							= scalar_mom[g+(ng*l)+(ng*(cmom-1)*i)+(ng*(cmom-1)*nx*j)+(ng*(cmom-1)*nx*ny*k)];
					}
				}
			}
		}
	}
}

// Copy the flux_out buffer back to the host
void ext_get_transpose_output_flux_(double* output_flux)
{
	double **tmp = (global_timestep % 2 == 0) ? flux_out : flux_in;

	// Transpose the data into the original SNAP format
	for (int a = 0; a < nang; a++)
	{
		for (int g = 0; g < ng; g++)
		{
			for (int i = 0; i < nx; i++)
			{
				for (int j = 0; j < ny; j++)
				{
					for (int k = 0; k < nz; k++)
					{
						for (int o = 0; o < noct; o++)
						{
							output_flux[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)] 
								= tmp[o][a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)];
						}
					}
				}
			}
		}
	}
}
