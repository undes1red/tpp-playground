import gc
import matplotlib.pyplot as plt
import os

def save_fig(fig, file_location, file_name):
    fig.savefig(os.path.join(file_location, file_name), bbox_inches = "tight")
    fig.clear()
    plt.close(fig = fig)
    del fig
    gc.collect()