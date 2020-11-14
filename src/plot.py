import matplotlib.pyplot as pyplot
import os


# def generate_plots(root_dir):



root_dir = '../logs'
for (dirpath, dirnames, filenames) in os.walk(root_dir):
    print(filenames)