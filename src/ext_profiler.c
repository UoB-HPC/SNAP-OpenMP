#include <stdio.h>
#include <string.h>
#include <time.h>
#include "ext_profiler.h"

#ifdef ENABLE_PROFILING

#define MS 1000.0
#define NS 1000000000.0
#define NS_MS 1000000.0
#define _PROFILER_MAX_KERNELS 2048

#pragma omp declare target

// Internal variables
struct timespec _profiler_start;
struct timespec _profiler_end;
unsigned int _profiler_kernelcount = 0;
profile _profiler_entries[_PROFILER_MAX_KERNELS];

#pragma omp end declare target

// Internally start the profiling timer
void _profiler_start_timer()
{
    clock_gettime(CLOCK_MONOTONIC, &_profiler_start);
}

// Internally end the profiling timer and store results
void _profiler_end_timer(const char* kernel_name)
{
    clock_gettime(CLOCK_MONOTONIC, &_profiler_end);

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
    long elapsed_sec = _profiler_end.tv_sec-_profiler_start.tv_sec;
    long elapsed_ns = _profiler_end.tv_nsec-_profiler_start.tv_nsec;
    _profiler_entries[ii].time += (elapsed_ns < 0) 
        ? (elapsed_sec-1)*MS + (NS+elapsed_ns)/NS_MS 
        : elapsed_sec*MS + elapsed_ns/NS_MS;
    _profiler_entries[ii].calls++;
}

// Print the profiling results to output
void _profiler_print_results()
{
    printf("\n*************************************************************\n");
    //for(int ii = 0; ii < 58; ++ii) printf("*");
    printf("\nProfiling Results:\n\n");
    printf("%-30s%8s%20s\n", "Kernel Name", "Calls", "Runtime (ms)");

    double total_elapsed_time;
    for(int ii = 0; ii < _profiler_kernelcount; ++ii)
    {
        total_elapsed_time += _profiler_entries[ii].time;

        printf("%-30s%8d%20.06F\n", _profiler_entries[ii].name, 
                _profiler_entries[ii].calls, _profiler_entries[ii].time);
    }

    printf("\nTotal elapsed time: %.06Fms\n", total_elapsed_time);
    printf("\n*************************************************************\n\n");
}

#endif

