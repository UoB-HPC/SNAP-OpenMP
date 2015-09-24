#include "ocl_sweep.h"

// Create buffers and copy the flux, source and
// cross section arrays to the OpenCL device
//
// Argument list:
// nx, ny, nz are the (local to MPI task) dimensions of the grid
// ng is the number of energy groups
// cmom is the "computational number of moments"
// ichunk is the number of yz planes in the KBA decomposition
// dd_i, dd_j(nang), dd_k(nang) is the x,y,z (resp) diamond difference coefficients
// mu(nang) is x-direction cosines
// scat_coef [ec](nang,cmom,noct) - Scattering expansion coefficients
// time_delta [vdelt](ng)              - time-absorption coefficient
// total_cross_section [t_xs](nx,ny,nz,ng)       - Total cross section on mesh
// flux_in(nang,nx,ny,nz,noct,ng)   - Incoming time-edge flux pointer
// denom(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
// weights(nang) - angle weights for scalar reduction
void copy_to_device_(
		double *mu, 
		double *eta, 
		double *xi,
		double *scat_coef,
		double *total_cross_section,
		double *weights,
		double *velocity,
		double *xs,
		int *mat,
		double *fixed_source,
		double *gg_cs,
		int *lma,
		double *g2g_source,
		double *flux_in)
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

	zero_edge = (double *)calloc(sizeof(double), s);
	flux_in = malloc(sizeof(double*)*noct);
	flux_out = malloc(sizeof(double*)*noct);

	// Allocate flux in and out
	for (unsigned int o = 0; o < noct; o++)
	{
		flux_in[o] = malloc(sizeof(double)*nang*nx*ny*nz*ng);
		flux_out[o] = malloc(sizeof(double)*nang*nx*ny*nz*ng);

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

	dd_j = malloc(sizeof(double)*nang);
	dd_k = malloc(sizeof(double)*nang);
	mu = malloc(sizeof(double)*nang);
	eta = malloc(sizeof(double)*nang);
	xi = malloc(sizeof(double)*nang);
	scat_coeff = malloc(sizeof(double)*nang*cmom*noct);
	total_cross_section = malloc(sizeof(double)*nx*ny*nz*ng);
	xs = malloc(sizeof(double)*nmat*ng);
	map = malloc(sizeof(int)*nx*ny*nz);
	fixed_source = malloc(sizeof(double)*nx*ny*nz*ng);
	gg_cs = malloc(sizeof(double)*nmat*nmom*ng*ng);
	lma = malloc(sizeof(double)*nmom);
	g2g_source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);

	if (cmom == 1)
	{
		scalar_mom = malloc(sizeof(double)*nx*ny*nz*ng);
	}
	else
	{
		scalar_mom = malloc(sizeof(double)*nx*ny*nz*ng);
	}

	zero_flux_moments_buffer();

	scat_cs = malloc(sizeof(double)*nmom*nx*ny*nz*ng);
	weights = malloc(sizeof(double)*nang);
	denom = malloc(sizeof(double)*nang*nx*ny*nz*ng);
	source = malloc(sizeof(double)*cmom*nx*ny*nz*ng);
	time_delta = malloc(sizeof(double)*ng);
	velocity = malloc(sizeof(double)*ng);
	scalar_flux = malloc(sizeof(double)*nx*ny*nz*ng);

	zero_scalar_flux();

	groups_todo = malloc(sizeof(unsigned int)*ng);
}

// Copy the scalar flux value back to the host and transpose
void get_scalar_flux_trans_(double *scalar)
{
	double *tmp = malloc(sizeof(double)*nx*ny*nz*ng);
	cl_int err;
	err = clEnqueueReadBuffer(queue[0], d_scalar_flux, CL_TRUE, 0, sizeof(double)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
	for (unsigned int g = 0; g < ng; g++)
		for (unsigned int i = 0; i < nx; i++)
			for (unsigned int j = 0; j < ny; j++)
				for (unsigned int k = 0; k < nz; k++)
					scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] = tmp[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)];

	free(tmp);
}

void get_scalar_flux_moments_(double *scalar_moments)
{
	double *tmp = malloc(sizeof(double)*(cmom-1)*nx*ny*nz*ng);
	cl_int err;
	err = clEnqueueReadBuffer(queue[0], d_scalar_mom, CL_TRUE, 0, sizeof(double)*(cmom-1)*nx*ny*nz*ng, tmp, 0, NULL, NULL);
	for (unsigned int g = 0; g < ng; g++)
		for (unsigned int l = 0; l < cmom-1; l++)
			for (unsigned int i = 0; i < nx; i++)
				for (unsigned int j = 0; j < ny; j++)
					for (unsigned int k = 0; k < nz; k++)
						scalar_moments[l+((cmom-1)*i)+((cmom-1)*nx*j)+((cmom-1)*nx*ny*k)+((cmom-1)*nx*ny*nz*g)] = tmp[g+(ng*l)+(ng*(cmom-1)*i)+(ng*(cmom-1)*nx*j)+(ng*(cmom-1)*nx*ny*k)];
	free(tmp);
}


// Copy the flux_out buffer back to the host
void get_output_flux_(double* flux_out)
{
	double *tmp = calloc(sizeof(double),nang*ng*nx*ny*nz*noct);
	cl_int err;
	for (unsigned int o = 0; o < noct; o++)
	{
		if (global_timestep % 2 == 0)
			err = clEnqueueReadBuffer(queue[0], d_flux_out[o], CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, &(tmp[nang*ng*nx*ny*nz*o]), 0, NULL, NULL);
		else
			err = clEnqueueReadBuffer(queue[0], d_flux_in[o], CL_TRUE, 0, sizeof(double)*nang*nx*ny*nz*ng, &(tmp[nang*ng*nx*ny*nz*o]), 0, NULL, NULL);
	}

	// Transpose the data into the original SNAP format
	for (int a = 0; a < nang; a++)
		for (int g = 0; g < ng; g++)
			for (int i = 0; i < nx; i++)
				for (int j = 0; j < ny; j++)
					for (int k = 0; k < nz; k++)
						for (int o = 0; o < noct; o++)
							flux_out[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)] = tmp[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*ng*nx*ny*nz*o)];
	free(tmp);
}
