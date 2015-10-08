#pragma once 

#ifndef OFFLOAD
    #define OFFLOAD 0
#endif
#ifndef MIC_DEVICE
    #define MIC_DEVICE 0
#endif

#define VEC_ALIGN 64

// Entry point for completing the solve
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
		int *lma);

// Entry point for initialising problem params
void initialise_parameters(
    int *nx_, int *ny_, int *nz_,
    int *ng_, int *nang_, int *noct_,
    int *cmom_, int *nmom_,
    int *ichunk_,
    double *dx_, double *dy_, double *dz_,
    double *dt_,
    int *nmat_,
    int *timesteps_, int *outers_, int *inners_,
    double *epsi_, double *tolr_);

// Allocates buffers on the device
void initialise_host_memory(
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
		int *lma_in);

// Do the timestep, outer and inner iterations
void iterate(void);

// Compute the scalar flux from the angular flux
void reduce_angular(void);

// Transposes the scattering coefficient matrix
double* transpose_scat_coeff(double* scat_coeff_in);

// Transposes the scalar flux back to SNAP format
void ext_get_transpose_scalar_flux_(double *scalar);

// Transposes the scalar moments back to SNAP format
void ext_get_transpose_scalar_moments_(double *scalar_moments);

// Transposes the output flux back to SNAP format
void ext_get_transpose_output_flux_(double* output_flux);


