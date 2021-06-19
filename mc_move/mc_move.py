import numpy as np
import random

from pair_potential import lj_energy_one_particle_c_hard,lj_energy_one_partivle_c_periodic

def random_move(configuration,L,delta,random_selected,boundary_periodic,dimension=2,sigma=0.1):
    """
    make random move
    and handle the boundary condition
    and overlap
    """

    if boundary_periodic == True:

        config_copy = configuration.copy()
        for d in range(dimension):
            x_new = (configuration[dimension*random_selected + d] 
                                                        + (random.random() - 0.5) * delta)
            if x_new > L:   #handle if the new x is out of box
                x_new %= L
            elif x_new < 0:
                x_new %= L
            config_copy[dimension*random_selected + d] = x_new

    if boundary_periodic == False:
        config_copy = configuration.copy()
        for d in range(dimension):
            x_new = (configuration[dimension*random_selected + d] 
                                                        + (random.random() - 0.5) * delta)
            if x_new > L: #handle if the new x is out of box
                x_new = L - (x_new % L)
            elif x_new < 0:
                x_new = L - (x_new % L)
            config_copy[dimension*random_selected + d] = x_new

    overlap = False
    particles = configuration.size // dimension
    for par_i in range(particles):
        if par_i == random_selected:
            continue
        else:
            r2_judge = 0
            for d in range(dimension):
                delta_i = abs(config_copy[dimension * random_selected + d] - config_copy[par_i*dimension + d])
                r2_judge += delta_i**2
            
            if r2_judge < sigma:
                overlap = True
                break

    return config_copy,overlap





def mc_move(configuration,L,delta,boundary_periodic,temperature,dimension = 2):
    """
    make a random move in the configuration
    """

    new_config_accepted = False 
    beta = 1/temperature   
    total_particle = configuration.size // dimension
    random_selected = int(random.random() * total_particle)
    overlap = False

    if boundary_periodic == True:
        energy_before_move = lj_energy_one_partivle_c_periodic(configuration,L,random_selected)
    elif boundary_periodic == False:
        energy_before_move = lj_energy_one_particle_c_hard(configuration,random_selected)
    
    new_configuration,overlap = random_move(configuration,L,delta,random_selected,boundary_periodic)
    while overlap == True:
        new_configuration,overlap = random_move(configuration,L,delta,random_selected,boundary_periodic)

    if boundary_periodic == True:
        energy_after_move = lj_energy_one_partivle_c_periodic(new_configuration,L,random_selected)
    elif boundary_periodic == False:
        energy_after_move = lj_energy_one_particle_c_hard(new_configuration,random_selected)

    energy_difference = energy_after_move - energy_before_move

    if random.random() < np.exp(-beta * energy_difference):
        new_config_accepted = True
        configuration = new_configuration
    
    return configuration,energy_difference,new_config_accepted

#%%
# %%
