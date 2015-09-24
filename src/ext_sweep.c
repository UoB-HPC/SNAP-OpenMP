#include "ocl_sweep.h"

// Forward declare to zero buffer functions
extern void zero_edge_flux_buffers_(void);
extern void zero_centre_flux_in_buffer_(void);

struct cell 
{
    unsigned int i,j,k;
};

typedef struct 
{
    unsigned int num_cells;
    struct cell *cells;

    // index is an index into the cells array for when storing the cell indexes
    unsigned int index;
} plane;

// Compute the order of the sweep for the first octant
plane *compute_sweep_order(void)
{
    unsigned int nplanes = ichunk + ny + nz - 2;
    plane *planes = (plane *)malloc(sizeof(plane)*nplanes);
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].num_cells = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                planes[n].num_cells++;
            }
        }
    }

    // Allocate the memory for each plane
    for (unsigned int i = 0; i < nplanes; i++)
    {
        planes[i].cells = (struct cell *)malloc(sizeof(struct cell)*planes[i].num_cells);
        planes[i].index = 0;
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                unsigned int idx = planes[n].index;
                planes[n].cells[idx].i = i;
                planes[n].cells[idx].j = j;
                planes[n].cells[idx].k = k;
                planes[n].index += 1;
            }
        }
    }

    return planes;
}

// Sweep over the grid and compute the angular flux
void sweep_octant(const unsigned int timestep, const unsigned int oct, const unsigned int ndiag, const plane *planes, const unsigned int num_groups_todo)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    int order_3d = 0b000001010011100101110111;

    int order_2d = 0b11100100;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;

	for (unsigned int d = 0; d < ndiag; d++)
	{
		sweep_cell(istep, jstep, kstep, oct, cell_index, groups_todo, num_groups_todo, planes[d].num_cells);
	}
}

// Perform a sweep over the grid for all the octants
void ext_sweep_(unsigned int num_groups_todo)
{
	// Number of planes in this octant
	unsigned int ndiag = ichunk + ny + nz - 2;

	// Get the order of cells to enqueue
	plane *planes = compute_sweep_order();

	// Set the constant kernel arguemnts
	set_sweep_cell_args();

	for (int o = 0; o < noct; o++)
	{
		enqueue_octant(global_timestep, o, ndiag, planes, num_groups_todo);
		zero_edge_flux_buffers_();
	}

	// Free planes
	for (unsigned int i = 0; i < ndiag; i++)
	{
		free(planes[i].cells);
	}

	free(planes);
}

