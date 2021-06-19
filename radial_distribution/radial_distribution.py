import numpy as np

def radial_distribute(configuration,L,rho,r_count,dimension=2):
    """
    calculate the radial distribution 
    with periodic boundary condition
    from any of the particle(since periodic, g(r) from any particle is the same)

    param ndarray configuration: configuration of the particles
    param float rho: given density
    param int L: the box's length
    param int r_count: the sample point number of r

    return ndarray g_r: the radial distribution g(r)
    return ndarray r_list: corresponding r index with g(r)
    """

    config_use = configuration.copy()
    particles = config_use.size // dimension
    r_max = np.sqrt(2) * L + 1
    r_list = np.linspace(0,r_max,r_count)
    count_at_r = np.zeros_like(r_list)
    g_r = np.zeros_like(r_list)
    dist_list = []

    x_0_index = 0

    for x_i in range(1,particles):
        r_i = 0
        for d in range(dimension):
            delta_x = abs(config_use[x_0_index*dimension + d] - config_use[x_i*dimension + d])
            if delta_x > L/2:
                delta_x = abs(L - delta_x)
            r_i += delta_x * delta_x
        dist_list.append(r_i)

    dist_list = np.array(dist_list)
    dist_list = np.sqrt(dist_list)

    for r_i in dist_list:
        count = 0
        r_j = r_list[count]
        while r_i > r_j:
            count += 1
            r_j = r_list[count]
        count_at_r[count-1] += 1

    count = 0
    while count < r_count-1:
        r_i = r_list[count]
        r_j = r_list[count+1]
        # d_r = r_j - r_i
        d_v = np.pi * (r_j**2 - r_i**2)
        par_in_dv = count_at_r[count]
        ideal_in_dv = d_v * rho
        g_r[count] = par_in_dv / (ideal_in_dv)
        count += 1

    return g_r, r_list


    

            

    