import cython
import numpy as np

DTYPE = np.float


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_c(double [::1,] x, double sigma=1.0, double epsilon=1.0, int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double energy = 0.0
    cdef double rija, r2, r6

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            for d in range(dimensions):
                r2 += (x[dimensions*i+d]-x[dimensions*j+d])*(x[dimensions*i+d]-x[dimensions*j+d])
            
            r6 = 1.0/r2/r2/r2
            if r2 > 2.5*2.5:
                energy += 0
            else:
                energy += 4.0*epsilon*r6*(r6 - 1.0)
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_gradient_c(double [::1,] x, double sigma=1.0, double epsilon=1.0, int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double rija, r2, r6, gij, gia

    gradient = np.zeros_like(x)
    cdef double[::1] gradient_view = gradient

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            gij = 0.0
            for d in range(dimensions):
                rija = x[dimensions*i+d]-x[dimensions*j+d]
                r2 += rija*rija
            
            if r2 <= 2.5*2.5:
                r6 = 1.0/r2/r2/r2
                gij = 24.0*epsilon*r6*(1.0-2.0*r6)/r2
            else:
                gij = 0

            for d in range(dimensions):
                gia = gij*(x[dimensions*i+d]-x[dimensions*j+d])
                gradient_view[dimensions*i+d] += gia
                gradient_view[dimensions*j+d] -= gia

    return gradient



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_c_periodic(double [::1,] x, double L, double sigma=1.0, double epsilon=1.0, double r_c = 2.5,int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double energy = 0.0
    cdef double rija, r2, r6, delta_x

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            for d in range(dimensions):
                delta_x = abs(x[dimensions*i+d]-x[dimensions*j+d])
                if delta_x > r_c:
                    delta_x = L - delta_x
                    
                r2 += delta_x*delta_x
            if r2 == 0:
                raise Warning("r2=0, overlap occured")
            
            if r2 > r_c*r_c:
                energy += 0
            else:
                r6 = 1.0/r2/r2/r2   
                energy += 4.0*epsilon*r6*(r6 - 1.0)

    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_gradient_c_periodic(double [::1,] x, double L, double sigma=1.0, double epsilon=1.0, double r_c = 2.5,int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double rija, r2, r6, gij, gia

    gradient = np.zeros_like(x)
    cdef double[::1] gradient_view = gradient

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            gij = 0.0
            for d in range(dimensions):
                rija = abs(x[dimensions*i+d]-x[dimensions*j+d])
                if rija > r_c:
                    rija = L - rija
                r2 += rija*rija

            if r2 == 0:
                raise Warning("r2=0, overlap occured")       

            if r2 <= r_c*r_c:
                r6 = 1.0/r2/r2/r2
                gij = 24.0*epsilon*r6*(1.0-2.0*r6)/r2
            else:
                gij = 0

            for d in range(dimensions):
                gia = gij*(x[dimensions*i+d]-x[dimensions*j+d])
                gradient_view[dimensions*i+d] += gia
                gradient_view[dimensions*j+d] -= gia

    return gradient



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_one_particle_c_hard(double [::1,] x, int particle_index, double sigma=1.0, double epsilon=1.0, int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double energy = 0.0
    cdef double rija, r2, r6

    for i in range(particles):
        if i != particle_index:
            r2 = 0.0
            for d in range(dimensions):
                r2 += (x[dimensions*i+d]-x[dimensions*particle_index+d])*(x[dimensions*i+d]-x[dimensions*particle_index+d])
            if r2 == 0:
                raise Warning("overlap encountered! \n "
                            "Please check!")
            r6 = 1.0/r2/r2/r2
            if r2 > 2.5*2.5:
                energy += 0
            else:
                energy += 4.0*epsilon*r6*(r6 - 1.0)
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_one_partivle_c_periodic(double [::1,] x, double L, int particle_index, double sigma=1.0, double epsilon=1.0, double r_c = 2.5,int dimensions=2):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double energy = 0.0
    cdef double rija, r2, r6, delta_x

    for i in range(particles):
        if i != particle_index:
            r2 = 0.0
            for d in range(dimensions):
                delta_x = abs(x[dimensions*i+d]-x[dimensions*particle_index+d])

                if delta_x > r_c:
                    delta_x = L - delta_x
                    
                r2 += delta_x*delta_x
            if r2 == 0:
                    raise Warning("overlap encountered! \n "
                                "Please check!")
            if r2 > r_c*r_c:
                energy += 0
            else:
                r6 = 1.0/r2/r2/r2   
                energy += 4.0*epsilon*r6*(r6 - 1.0)

    return energy