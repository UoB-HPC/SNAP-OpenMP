#include <stdio.h>
#include <string.h>
#include "ext_profiler.h"
#include "omp.h"

#ifdef ENABLE_PROFILING

#define _PROFILER_MAX_KERNELS 2048

#pragma omp declare target

// Internal variables
double _profiler_start;
double _profiler_end;
unsigned int _profiler_kernelcount = 0;
profile _profiler_entries[_PROFILER_MAX_KERNELS];

#pragma omp end declare target

// Internally start the profiling timer
void _profiler_start_timer()
{
    _profiler_start = omp_get_wtime();
}

// Internally end the profiling timer and store results
void _profiler_end_timer(const char* kernel_name)
{
    _profiler_end = omp_get_wtime();

    // Check if an entry exists
    int ii;
    for(ii = 0; ii < _profiler_kernelcount; ++ii)
    {
        if(!strcmp(_profiler_entries[ii].name, kernel_name))
        {
            break;
        }
    }

    // Create new entry
    if(ii == _profiler_kernelcount)
    {
        _profiler_kernelcount++;
        strcpy(_profiler_entries[ii].name,kernel_name);
    }

    // Update number of calls and time
    _profiler_entries[ii].time += _profiler_end-_profiler_start;
    _profiler_entries[ii].calls++;
}

// Print the profiling results to output
void _profiler_print_results()
{
    printf("\n-------------------------------------------------------------\n");
    printf("\nProfiling Results:\n\n");
    printf("%-30s%8s%20s\n", "Kernel Name", "Calls", "Runtime (s)");

    double total_elapsed_time = 0.0;
    for(int ii = 0; ii < _profiler_kernelcount; ++ii)
    {
        total_elapsed_time += _profiler_entries[ii].time;

        printf("%-30s%8d%20.03F\n", _profiler_entries[ii].name, 
                _profiler_entries[ii].calls, 
                _profiler_entries[ii].time);
    }

    printf("\nTotal elapsed time: %.03Fs, entries * are excluded.\n", total_elapsed_time);
    printf("\n-------------------------------------------------------------\n\n");
}

#endif

