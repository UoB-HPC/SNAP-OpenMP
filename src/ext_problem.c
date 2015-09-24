#include "ocl_problem.h"

#include <stdio.h>

void set_ocl_problem_(
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
        printf("Warning: nx and ichunk are different - expect the answers to be wrong...\n");
}
