#ifndef __SHARED
#define __SHARED

/*
 *		Array access macros
 */
#define flux_out(a,g,i,j,k) flux_out[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define flux_in(a,g,i,j,k) flux_in[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
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
#define scat_coef(a,m,o) scat_coef[(a)+(nang*(m))+(nang*cmom*(o))]
#define time_delta(g) time_delta[(g)]
#define total_cross_section(g,i,j,k) total_cross_section[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define scalar(g,i,j,k) scalar[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define weights(a) weights[(a)]

#define angular0(a,g,i,j,k) angular0[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular1(a,g,i,j,k) angular1[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular2(a,g,i,j,k) angular2[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular3(a,g,i,j,k) angular3[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular4(a,g,i,j,k) angular4[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular5(a,g,i,j,k) angular5[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular6(a,g,i,j,k) angular6[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular7(a,g,i,j,k) angular7[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define angular_prev0(a,g,i,j,k) angular_prev0[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev1(a,g,i,j,k) angular_prev1[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev2(a,g,i,j,k) angular_prev2[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev3(a,g,i,j,k) angular_prev3[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev4(a,g,i,j,k) angular_prev4[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev5(a,g,i,j,k) angular_prev5[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev6(a,g,i,j,k) angular_prev6[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define angular_prev7(a,g,i,j,k) angular_prev7[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]

#define velocity(g) velocity[(g)]

#define map(i,j,k) map[(i)+(nx*(j))+(nx*ny*(k))]
#define xs(i,g) xs[(i)+(nmat*(g))]

#define g2g_source(m,i,j,k,g) g2g_source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define fixed_source(i,j,k,g) fixed_source[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define gg_cs(m,l,g1,g2) gg_cs[(m)+(nmat*(l))+(nmat*nmom*(g1))+(nmat*nmom*ng*(g2))]
#define lma(m) lma[(m)]
#define scalar_mom(g,m,i,j,k) scalar_mom[(g)+((ng)*(m))+(ng*(cmom-1)*(i))+(ng*(cmom-1)*nx*(j))+(ng*(cmom-1)*nx*ny*(k))]

#define scat_cs(m,i,j,k,g) scat_cs[(m)+(nmom*(i))+(nmom*nx*(j))+(nmom*nx*ny*(k))+(nmom*nx*ny*nz*(g))]

struct cell {
    unsigned int i,j,k;
};

#endif
