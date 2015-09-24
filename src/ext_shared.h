#pragma once

/*
 *		Array access macros
 */
#define l_flux_out(a,g,i,j,k) l_flux_out[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define l_flux_in(a,g,i,j,k) l_flux_in[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define source(m,i,j,k,g) source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define flux_i(a,g,j,k) flux_i[(a)+(nang*(g))+(nang*ng*(j))+(nang*ng*ny*(k))]
#define flux_j(a,g,i,k) flux_j[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(k))]
#define flux_k(a,g,i,j) flux_k[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))]
#define denom(a,g,i,j,k) denom[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define dd_j(a) dd_j[(a)]
#define dd_k(a) dd_k[(a)]
#define mu(a) mu[(a)]
#define eta(a) eta[(a)]
#define xi(a) xi[(a)]
#define scat_coeff(a,m,o) scat_coeff[(a)+(nang*(m))+(nang*cmom*(o))]
#define time_delta(g) time_delta[(g)]
#define total_cross_section(g,i,j,k) total_cross_section[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define scalar_flux(g,i,j,k) scalar_flux[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define weights(a) weights[(a)]

#define angular(o,a,g,i,j,k) angular[o][(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev(o,a,g,i,j,k) angular_prev[o][(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define velocity(g) velocity[(g)]

#define map(i,j,k) map[(i)+(nx*(j))+(nx*ny*(k))]
#define xs(i,g) xs[(i)+(nmat*(g))]

#define g2g_source(m,i,j,k,g) g2g_source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define fixed_source(i,j,k,g) fixed_source[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define gg_cs(m,l,g1,g2) gg_cs[(m)+(nmat*(l))+(nmat*nmom*(g1))+(nmat*nmom*ng*(g2))]
#define lma(m) lma[(m)]
#define scalar_mom(g,m,i,j,k) scalar_mom[(g)+((ng)*(m))+(ng*(cmom-1)*(i))+(ng*(cmom-1)*nx*(j))+(ng*(cmom-1)*nx*ny*(k))]

#define scat_cs(m,i,j,k,g) scat_cs[(m)+(nmom*(i))+(nmom*nx*(j))+(nmom*nx*ny*(k))+(nmom*nx*ny*nz*(g))]
