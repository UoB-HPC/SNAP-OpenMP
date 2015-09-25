#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

// This file contains a list of the global problem variables
// such as grid size, angles, energy groups, etc.
int nx;
int ny;
int nz;
int ng;
int nang;
int noct;
int cmom;
int nmom;
int nmat;

int ichunk;
int timesteps;

double dt;
double dx;
double dy;
double dz;

int outers;
int inners;

double epsi;
double tolr;

// Data
double* source;
double** flux_in;
double** flux_out;
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
int* map;
double* fixed_source;
double* gg_cs;
double* lma;
double* g2g_source;
double* scalar_mom;
double* scat_cs;
int* groups_todo;

// Global variable for the timestep
unsigned int global_timestep;
