import numpy as np

def overlap_judge(configuration,dimension=2,sigma=1.0):
    """
    param ndarray configuration: the configuration of the particles
    return Boolean overlap: True for overlap occurs
                            False otherwise
    """
    config_judge = np.array(configuration)
    particles = config_judge.size // 2
    
    if particles == 1:
        return False
    else:
        overlap = False
        for par_i in range(particles):
            for par_j in range(par_i+1,particles):
                r2 = 0
                for d in range(dimension):
                    delta_i = config_judge[par_i*dimension + d] - config_judge[par_j*dimension + d]
                    r2 += delta_i**2

                if r2 < sigma**2:
                    overlap = True

                if overlap == True:
                    break
            if overlap == True:
                break
        return overlap


if __name__ == "__main__":
    overlap = False
    config = np.array([0,0,
                       1,0,
                       1,1,
                       1/2,1/2])

    overlap = overlap_judge(config)

    print(f"overlap={overlap}")