import numpy as np
import random

from pair_potential import lj_gradient_c,lj_gradient_c_periodic
from generate_config import gauss_distributed
from mc_move import mc_move


def motion_integrate(configuration,velocities,forces,
                    dt,temperature,nu,boundary_periodic,L,
                    func_mode, dimension=2,sigma=1.0):
    """
    integration equations of motion
    with Anderson thermostat
    using velocity Verlet method

    param ndarray configuration: old configuration of the particles
    param ndarray velocities: old velocities of the particles
    param ndarray forces: the forces calculated with the old configuration
    param float dt: time interval
    param float temperature: the simulation temperature
    param float nu: collision frequency
    param boolean boundary_periodic: True for periodic boundary condition
    param float L: the size of the box
    param int func_mode: 1 for first update and 2 for second update

    return ndarray configuration: new configuration of the particles
    return ndarray velocities: new velocities of the particles    
    """
    particle_count = configuration.size // dimension
    particle_list = np.linspace(0,particle_count-1,particle_count,dtype=int)
    new_config = np.zeros_like(configuration)
    new_velocity = np.zeros_like(configuration)

    if func_mode == 1:
        for par_i in particle_list:
            for d in range(dimension):
                new_config[dimension*par_i + d] = (configuration[dimension*par_i + d] 
                                                    + dt * velocities[dimension*par_i + d]
                                                    + dt * dt * forces[dimension*par_i + d] / 2)
                new_velocity[dimension*par_i + d] = (velocities[dimension*par_i + d]
                                                    + dt * forces[dimension*par_i + d] / 2)
                #deal with overlap and out-of-the-box
                #out of the box
                if new_config[dimension*par_i + d] > L :
                    if boundary_periodic == True:
                        new_config[dimension*par_i + d] =  (new_config[dimension*par_i + d] % L)
                    elif boundary_periodic == False:
                        new_config[dimension*par_i + d] = L - (new_config[dimension*par_i + d] % L)
                elif new_config[dimension*par_i + d] < 0:
                    if boundary_periodic == True:
                        new_config[dimension*par_i + d] =  (new_config[dimension*par_i + d] % L)
                    elif boundary_periodic == False:
                        new_config[dimension*par_i + d] =  ( new_config[dimension*par_i + d] % L)        

        #overlap
        overlap = False
        for i in range(particle_count):
            for j in range(i+1,particle_count):
                r2 = 0.0
                for d in range(dimension):
                    delta_x = abs(new_config[dimension*i+d]-new_config[dimension*j+d])                    
                    r2 += delta_x*delta_x
                if r2 < sigma:
                    overlap = True

        if overlap == True:
            new_config,new_velocity = motion_integrate(configuration,velocities,forces,dt/1000,temperature,nu,boundary_periodic,L,func_mode=1)

    elif func_mode == 2:
        for par_i in particle_list:
            for d in range(dimension):
                new_velocity[dimension*par_i + d] = (velocities[dimension*par_i + d]
                                                    + dt * forces[dimension*par_i + d] / 2)
        
        #Anderson heat bath
        sigma = np.sqrt(temperature)
        for par_i in particle_list:
            for d in range(dimension):
                if random.random() < nu * dt:
                    new_velocity[dimension*par_i + d] = gauss_distributed(sigma)

    return new_config,new_velocity




def md_move(configuration,velocities,forces,boundary_periodic,dt,L,temperature,nu,args=()):
    """
    perform one step of MD simulation

    param ndarray configuration: old configuration of the particles
    param ndarray velocities: old velocities of the particles
    param ndarray forces: the forces calculated with the old configuration
    param boolean boundary_periodic: True for periodic boundary condition
    param float dt: time interval
    param float L: the size of the box
    param float temperature: the simulation temperature
    param float nu: collision frequency
    param list args: for gradient calculation, empty for hard boundary
                    and [L,r_c] for periodic boundary

    return ndarray configuration: new configuration of the particles
    return ndarray velocities: new velocities of the particles
    return ndarray forces: new forces of the particles
    """

    if boundary_periodic == True:
        gradient_function = lj_gradient_c_periodic
    elif boundary_periodic == False:
        gradient_function = lj_gradient_c

    new_config,new_velocity = motion_integrate(configuration,velocities,forces,
                                                dt,temperature,nu,
                                                boundary_periodic,L,func_mode=1)
    new_force = gradient_function(new_config,*args)
    _,new_velocity = motion_integrate(configuration,new_velocity,new_force,
                                    dt,temperature,nu,boundary_periodic,L,func_mode=2)

    return new_config,new_velocity,new_force

    

    