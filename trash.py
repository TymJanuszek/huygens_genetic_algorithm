import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from spherical import *
import time

epochs = 10
out_img_size = 100

pixel_size = 0.4

plate_div = 1
plate_size = int(out_img_size / plate_div)

# ___________________________________________________________________
new_img_2d = np.zeros(shape=(out_img_size, out_img_size))
best_img = 255 * np.ones(out_img_size * out_img_size, dtype=np.uint8)

goal_img = load_image('source_files/einstein_100x100.png')
# goal_img = load_image('source_files/pic.png')

empty_img = np.zeros(out_img_size ** 2)
empty = np.sum(np.abs(empty_img - goal_img))
print("fill:", str(round(100 - fit_percent(empty), 4)) + "%")
print("_______________________________________")

# ___________________________ single puntcure ___________________________________

best_fit = empty
fit_arr = np.zeros(epochs)
plate = random_plate(0, 10, plate_size)

best_plate = random_plate(0, 10, plate_size)
best_gen = 0

for gen in range(epochs):
    # for i in range(int(img_size ** 2 / epochs /100)):
    #     plate = puncture(plate)

    plate = puncture(plate, plate_size)

    new_img_2d = np.zeros(shape=(out_img_size, out_img_size))
    new_img_2d = simulate(new_img_2d, plate)

    new_img = 255 * new_img_2d / np.max(new_img_2d)
    new_img = np.round(np.abs(new_img.flatten()))
    new_fit = np.sum(np.abs(new_img - goal_img))
    fit_arr[gen] = fit_percent(new_fit)

    print(gen, "generation fitness: ", fit_percent_str(new_fit, 4))

    if new_fit < best_fit:
        best_img = new_img
        print(" new best!")
        best_plate = np.copy(plate)
        best_fit = new_fit
        best_gen = gen


    plate = np.copy(best_plate)
# ___________________________ wall throw algorithm ___________________________________

best_fit = 255 * out_img_size ** 2
best_plate = random_plate(0, 100)
fit_arr = np.zeros(epochs)

for gen in range(epochs):
    best_fit = fitness_func(best_img, goal_img)
    plate = random_plate(1, 1000)

    new_img_2d = np.zeros(shape=(out_img_size, out_img_size))
    new_img_2d = simulate(new_img_2d, plate)

    new_img = 255 * new_img_2d / np.max(new_img_2d)
    new_img = np.round(abs(new_img.flatten()))

    new_fit = fitness_func(new_img, goal_img)
    fit_arr[gen] = new_fit
    print(gen, "generation fitness: ", new_fit)

    if new_fit < best_fit:
        best_img = new_img
        print(" new best:", new_fit)
        best_plate = plate
        best_gen = gen