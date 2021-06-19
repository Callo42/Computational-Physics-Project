import matplotlib.pyplot as plt        
    
def plot_config(config_in,rho,L,point_size = 10):

    config = config_in
    config_x = config[::2]
    config_y = config[1::2]

    print(f"config with rho={rho} \n L={L}")

    figure, ax = plt.subplots()
    ax.scatter(config_x,config_y,s=point_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    ax.set_title("config generation")
        

    plt.show()