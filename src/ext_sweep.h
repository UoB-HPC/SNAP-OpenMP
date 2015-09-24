#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#include "ext_problem.h"

double* source;
double* *flux_in;
double* *flux_out;
double* flux_i;
double* flux_j;
double* flux_k;
double* denom;
double dd_i;
double* dd_j;
double* dd_k;
double* mu;
double* eta;
double* xi;
double* scat_coeff;
double* time_delta;
double* total_cross_section;
double* weights;
double* velocity;
double* scalar_flux;
double* xs;
double* map;
double* fixed_source;
double* gg_cs;
double* lma;
double* g2g_source;
double* scalar_mom;
double* scat_cs;
double* groups_todo;

// Create an empty buffer to zero out the edge flux arrays
// Each direction can share it as we make sure that it is
// big enough for each of them
double *zero_edge;

// Global variable for the timestep
unsigned int global_timestep;

void ocl_sweep_(unsigned int);
