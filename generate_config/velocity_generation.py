import numpy as np
import random

def gauss_distributed(sigma,mu=0):
    """
    generate gaussian distributed l
    with mean mu and standard deviation sigma
    """

    r = 2
    while r > 1:
        v_1 = 2 * random.random() - 1
        v_2 = 2 * random.random() - 1
        r = v_1 * v_1 + v_2 * v_2
    l = v_1 * np.sqrt(-2 * np.log(r) / r )
    l = mu + sigma * l 
    
    return l


def velocity_init(positions,temp,dimension=2):
    """
    generate initial velocity distibution
    with Maxwell distribution and temperature = temp
    reciving parameter positions: ndarray
    (x1,y1,x2,y2, ... , x_N,y_N)
    """
    position_size = positions.size
    N_list = list(range(int(position_size / dimension)))
    velocities = np.zeros_like(positions)
    sigma = np.sqrt(temp)    

    for particle_i in N_list:
        x_index = 2 * particle_i
        y_index = 2 * particle_i + 1
        velocities[x_index] = gauss_distributed(sigma)
        velocities[y_index] = gauss_distributed(sigma)
    
    return velocities




if __name__ == "__main__":
    #test the gauss_distributed function
    import matplotlib.pyplot as plt

    sample_points = 10000
    count = 0
    v_list = np.zeros(sample_points)

    T = 300

    sigma = np.sqrt(T)
    while count < sample_points:
        v_list[count] = gauss_distributed(sigma)
        count += 1

    v_max = v_list[0]
    for v in v_list:
        if abs(v) > v_max:
            v_max = abs(v)
    v_max += 0.1 * v_max
    axis_v = np.linspace(-v_max,v_max,int(sample_points/100))
    count_v = np.zeros_like(axis_v)
    for v in v_list:
        count_temp = 0
        while v > axis_v[count_temp]:
            count_temp += 1
        count_v[count_temp] += 1
    count_v /= sample_points

    figure,ax = plt.subplots()
    ax.scatter(axis_v,count_v,s=8)
    ax.set_xlabel("v")
    ax.set_ylabel("P(v)")
    plt.show()
        




