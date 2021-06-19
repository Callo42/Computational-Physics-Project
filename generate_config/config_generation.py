import numpy as np
import random
import matplotlib.pyplot as plt

from overlap_judge import overlap_judge

#%%
def get_par_rand(L,dimension=2):

    for d in range(dimension):
        x_new = random.random() * L
        y_new = random.random() * L
    return x_new,y_new



def generate_random(rho,L,dimension=2,sigma_for_overlap_check=0.5):
    """
    generate N particle coordinates
    to be used in lj_potential
    Here rho = N/L**2
    And make sure that configurations is getting rid of overlap
    """
    area = L*L
    N = int(area * rho)
    position = []
    sigma_for_overlap_check = sigma_for_overlap_check

    for par_i in range(N):
        overlap = False

        x_new,y_new = get_par_rand(L)
        position.append(x_new)
        position.append(y_new) 

        overlap = overlap_judge(position,sigma=sigma_for_overlap_check)
        
        while overlap == True:
            x_new,y_new = get_par_rand(L)
            position[par_i*dimension+0] = x_new
            position[par_i*dimension+1] = y_new

            overlap = overlap_judge(position,sigma=sigma_for_overlap_check)

    position = np.array(position)
    return position


def generate_lattice(rho,L,dimension=2):
    """
    generate N particle coordinates
    with square crystal lattice
    to be used in lj_potential
    Here rho = N/L**2
    And make sure that configurations is getting rid of overlap
    """

    area = L*L
    N = int(area * rho)
    total_number = dimension * N
    position = np.zeros(total_number)

    N_root = int(np.sqrt(N))+1
    axis_interval = L/(np.sqrt(N))

    partical_count = 0

    # 2D situation!
    while partical_count < N:
        x_index = 2 * partical_count
        y_index = x_index + 1
        position[x_index] = ( int(x_index/2) // N_root) * axis_interval
        position[y_index] = (int(y_index-1)/2 % N_root) * axis_interval
        partical_count += 1


    return position


if __name__ == '__main__':
    
    rho = 0.8
    L = 10
    point_size = 5

    config_random = generate_random(rho,L)
    config_square = generate_lattice(rho,L)

    random_x = config_random[::2]
    random_y = config_random[1::2]
    lattice_x = config_square[::2]
    lattice_y = config_square[1::2]

    print(f"random config with rho={rho} \n L={L}"
        f"\n is {config_random}")
    print(f"random_x = {random_x} \n random_y = {random_y}")    
    
    print(f"square lattice config with rho={rho} \n L={L}"
        f"\n is {config_square}")
    print(f"lattice_x = {lattice_x} \n lattice_y = {lattice_y}")
    
    figure, ax = plt.subplots()
    ax.scatter(random_x,random_y,s=point_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    ax.set_title("random generation")
    

    figure, ax = plt.subplots()
    ax.scatter(lattice_x,lattice_y,s=point_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    ax.set_title("square lattice generation")
    

    plt.show()



    