
#include "ocl_sweep.h"

// Enqueue the kernel to reduce the angular flux to the scalar flux
void ocl_scalar_flux_(void)
{
    cl_int err;

    const size_t global[3] = {nx, ny, nz};

    err = clSetKernelArg(k_reduce_angular, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_reduce_angular, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_reduce_angular, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_reduce_angular, 3, sizeof(unsigned int), &nang);
    err |= clSetKernelArg(k_reduce_angular, 4, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_reduce_angular, 5, sizeof(unsigned int), &noct);
    err |= clSetKernelArg(k_reduce_angular, 6, sizeof(unsigned int), &cmom);

    err |= clSetKernelArg(k_reduce_angular, 7, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(k_reduce_angular, 8, sizeof(cl_mem), &d_scat_coeff);

    if (global_timestep % 2 == 0)
    {
        err |= clSetKernelArg(k_reduce_angular, 9, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_angular, 10, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_angular, 11, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_angular, 12, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_angular, 13, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_angular, 14, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_angular, 15, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_angular, 16, sizeof(cl_mem), &d_flux_out[7]);

        err |= clSetKernelArg(k_reduce_angular, 17, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_angular, 18, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_angular, 19, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_angular, 20, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_angular, 21, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_angular, 22, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_angular, 23, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_angular, 24, sizeof(cl_mem), &d_flux_in[7]);
    }
    else
    {
        err |= clSetKernelArg(k_reduce_angular, 9, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_angular, 10, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_angular, 11, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_angular, 12, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_angular, 13, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_angular, 14, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_angular, 15, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_angular, 16, sizeof(cl_mem), &d_flux_in[7]);

        err |= clSetKernelArg(k_reduce_angular, 17, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_angular, 18, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_angular, 19, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_angular, 20, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_angular, 21, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_angular, 22, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_angular, 23, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_angular, 24, sizeof(cl_mem), &d_flux_out[7]);
    }

    err |= clSetKernelArg(k_reduce_angular, 25, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_reduce_angular, 26, sizeof(cl_mem), &d_scalar_flux);
    err |= clSetKernelArg(k_reduce_angular, 27, sizeof(cl_mem), &d_scalar_mom);
    check_error(err, "Setting reduce_angular kernel arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_reduce_angular, 3, 0, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueue reduce_angular kernel");

    //err = clFinish(queue[0]);
    //check_error(err, "Finishing queue after reduce_angular kernel");

}

void reduce_angular_cells(void)
{
    cl_int err;

    // Each kernel is a cell
    // Assign each energy group to a work group, so reductions take place
    // only within each workgroup.
    // Set the local size to the maximum allowed.
    size_t size;
    err = clGetKernelWorkGroupInfo(k_reduce_angular_cell, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
    check_error(err, "Getting max work group size for kernel/device");

    // Do naive reduction if too small
    if (size < nang)
    {
        printf("Warning: using naive reduction\n");
        ocl_scalar_flux_();
        return;
    }

    // get smallest power of 2 greater than nang
    size_t power = 1 << (unsigned int)ceil(log2((double)nang));
    if (power < size) size = power;

    const size_t global[2] = {size * ng, nx*ny*nz};
    const size_t local[2] = {size, 1};

    err = clSetKernelArg(k_reduce_angular_cell, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_reduce_angular_cell, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_reduce_angular_cell, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_reduce_angular_cell, 3, sizeof(unsigned int), &nang);
    err |= clSetKernelArg(k_reduce_angular_cell, 4, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_reduce_angular_cell, 5, sizeof(unsigned int), &noct);
    err |= clSetKernelArg(k_reduce_angular_cell, 6, sizeof(unsigned int), &cmom);

    err |= clSetKernelArg(k_reduce_angular_cell, 7, sizeof(double)*size, NULL);


    err |= clSetKernelArg(k_reduce_angular_cell, 8, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(k_reduce_angular_cell, 9, sizeof(cl_mem), &d_scat_coeff);

    if (global_timestep % 2 == 0)
    {
        err |= clSetKernelArg(k_reduce_angular_cell, 10, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_angular_cell, 11, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_angular_cell, 12, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_angular_cell, 13, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_angular_cell, 14, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_angular_cell, 15, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_angular_cell, 16, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_angular_cell, 17, sizeof(cl_mem), &d_flux_out[7]);

        err |= clSetKernelArg(k_reduce_angular_cell, 18, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_angular_cell, 19, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_angular_cell, 20, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_angular_cell, 21, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_angular_cell, 22, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_angular_cell, 23, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_angular_cell, 24, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_angular_cell, 25, sizeof(cl_mem), &d_flux_in[7]);
    }
    else
    {
        err |= clSetKernelArg(k_reduce_angular_cell, 10, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_angular_cell, 11, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_angular_cell, 12, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_angular_cell, 13, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_angular_cell, 14, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_angular_cell, 15, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_angular_cell, 16, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_angular_cell, 17, sizeof(cl_mem), &d_flux_in[7]);

        err |= clSetKernelArg(k_reduce_angular_cell, 18, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_angular_cell, 19, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_angular_cell, 20, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_angular_cell, 21, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_angular_cell, 22, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_angular_cell, 23, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_angular_cell, 24, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_angular_cell, 25, sizeof(cl_mem), &d_flux_out[7]);
    }

    err |= clSetKernelArg(k_reduce_angular_cell, 26, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_reduce_angular_cell, 27, sizeof(cl_mem), &d_scalar_flux);
    check_error(err, "Setting reduce_angular_cell kernel arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_reduce_angular_cell, 2, 0, global, local, 0, NULL, NULL);
    check_error(err, "Enqueue reduce_angular_cell kernel");

}

void reduce_moments_cells(void)
{
    cl_int err;

    // Each kernel is a cell
    // Assign each energy group to a work group, so reductions take place
    // only within each workgroup.
    // Set the local size to the maximum allowed.
    size_t size;
    err = clGetKernelWorkGroupInfo(k_reduce_moments_cell, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
    check_error(err, "Getting max work group size for kernel/device");

    // Do naive reduction if too small
    if (size < nang)
    {
        printf("Warning: using naive reduction\n");
        ocl_scalar_flux_();
        return;
    }

    // get the closest power of 2 to nang
    size_t power_up = 1 << (unsigned int)ceil(log2((double)nang));
    size_t power_down = 1 << (unsigned int)(ceil(log2((double)nang))-1);
    if (power_up - nang < nang - power_down)
        size = power_up;
    else
        size = power_down;

    const size_t global[2] = {size * ng, nx*ny*nz};
    const size_t local[2] = {size, 1};

    err = clSetKernelArg(k_reduce_moments_cell, 0, sizeof(unsigned int), &nx);
    err |= clSetKernelArg(k_reduce_moments_cell, 1, sizeof(unsigned int), &ny);
    err |= clSetKernelArg(k_reduce_moments_cell, 2, sizeof(unsigned int), &nz);
    err |= clSetKernelArg(k_reduce_moments_cell, 3, sizeof(unsigned int), &nang);
    err |= clSetKernelArg(k_reduce_moments_cell, 4, sizeof(unsigned int), &ng);
    err |= clSetKernelArg(k_reduce_moments_cell, 5, sizeof(unsigned int), &noct);
    err |= clSetKernelArg(k_reduce_moments_cell, 6, sizeof(unsigned int), &cmom);

    err |= clSetKernelArg(k_reduce_moments_cell, 7, sizeof(double)*size, NULL);


    err |= clSetKernelArg(k_reduce_moments_cell, 8, sizeof(cl_mem), &d_weights);
    err |= clSetKernelArg(k_reduce_moments_cell, 9, sizeof(cl_mem), &d_scat_coeff);

    if (global_timestep % 2 == 0)
    {
        err |= clSetKernelArg(k_reduce_moments_cell, 10, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_moments_cell, 11, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_moments_cell, 12, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_moments_cell, 13, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_moments_cell, 14, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_moments_cell, 15, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_moments_cell, 16, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_moments_cell, 17, sizeof(cl_mem), &d_flux_out[7]);

        err |= clSetKernelArg(k_reduce_moments_cell, 18, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_moments_cell, 19, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_moments_cell, 20, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_moments_cell, 21, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_moments_cell, 22, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_moments_cell, 23, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_moments_cell, 24, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_moments_cell, 25, sizeof(cl_mem), &d_flux_in[7]);
    }
    else
    {
        err |= clSetKernelArg(k_reduce_moments_cell, 10, sizeof(cl_mem), &d_flux_in[0]);
        err |= clSetKernelArg(k_reduce_moments_cell, 11, sizeof(cl_mem), &d_flux_in[1]);
        err |= clSetKernelArg(k_reduce_moments_cell, 12, sizeof(cl_mem), &d_flux_in[2]);
        err |= clSetKernelArg(k_reduce_moments_cell, 13, sizeof(cl_mem), &d_flux_in[3]);
        err |= clSetKernelArg(k_reduce_moments_cell, 14, sizeof(cl_mem), &d_flux_in[4]);
        err |= clSetKernelArg(k_reduce_moments_cell, 15, sizeof(cl_mem), &d_flux_in[5]);
        err |= clSetKernelArg(k_reduce_moments_cell, 16, sizeof(cl_mem), &d_flux_in[6]);
        err |= clSetKernelArg(k_reduce_moments_cell, 17, sizeof(cl_mem), &d_flux_in[7]);

        err |= clSetKernelArg(k_reduce_moments_cell, 18, sizeof(cl_mem), &d_flux_out[0]);
        err |= clSetKernelArg(k_reduce_moments_cell, 19, sizeof(cl_mem), &d_flux_out[1]);
        err |= clSetKernelArg(k_reduce_moments_cell, 20, sizeof(cl_mem), &d_flux_out[2]);
        err |= clSetKernelArg(k_reduce_moments_cell, 21, sizeof(cl_mem), &d_flux_out[3]);
        err |= clSetKernelArg(k_reduce_moments_cell, 22, sizeof(cl_mem), &d_flux_out[4]);
        err |= clSetKernelArg(k_reduce_moments_cell, 23, sizeof(cl_mem), &d_flux_out[5]);
        err |= clSetKernelArg(k_reduce_moments_cell, 24, sizeof(cl_mem), &d_flux_out[6]);
        err |= clSetKernelArg(k_reduce_moments_cell, 25, sizeof(cl_mem), &d_flux_out[7]);
    }

    err |= clSetKernelArg(k_reduce_moments_cell, 26, sizeof(cl_mem), &d_time_delta);
    err |= clSetKernelArg(k_reduce_moments_cell, 27, sizeof(cl_mem), &d_scalar_mom);
    check_error(err, "Setting reduce_moments_cell kernel arguments");

    err = clEnqueueNDRangeKernel(queue[0], k_reduce_moments_cell, 2, 0, global, local, 0, NULL, NULL);
    check_error(err, "Enqueue reduce_moments_cell kernel");

}
