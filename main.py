import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import spherical as sp
import time
from copy import deepcopy

mut_epochs = 20
cross_epochs = 1
epochs = 10

img_size = 100

pixel_size = 0.4

plate_div = 1
plate_size = int(img_size / plate_div)

new_img_2d = np.zeros(shape=(img_size, img_size))
best_img = 255 * np.ones(img_size * img_size, dtype=np.uint8)

goal_img = sp.load_image('source_files/einstein_100x100.png')
# goal_img = load_image('source_files/pic.png')

empty_img = np.zeros(img_size ** 2)
empty = np.sum(np.abs(empty_img - goal_img))
print("fill:", str(round(100 - sp.fit_percent(empty, img_size), 4)) + "%")

start = time.time()
# _______________________ make models _____________
model1 = sp.HuygensModel(goal_img, 100, 100, '_first_')
model2 = sp.HuygensModel(goal_img, 100, 100, '_second_')

fit_arr1 = np.zeros(epochs * 2)
fit_arr2 = np.zeros(epochs * 2)

for epoch in range(epochs):
    # ___________________ one point __________________
    model1.one_point_algorithm(mut_epochs)
    model2.one_point_algorithm(mut_epochs)

    fit_arr1[2 * epoch] = sp.fit_percent(model1.best_fit, model1.img_size)
    fit_arr2[2 * epoch] = sp.fit_percent(model2.best_fit, model2.img_size)

    # ______________ cross _______

    model1, model2 = sp.cross_and_evaluate(model1, model2, epoch)

    fit_arr1[2 * epoch + 1] = sp.fit_percent(model1.best_fit, model1.img_size)
    fit_arr2[2 * epoch + 1] = sp.fit_percent(model2.best_fit, model2.img_size)

# ___________________________ run statistics ___________________________________

print("_______________________________________")

print("1\nruntime:", round(time.time() - start, 2), "s")
print("exec speed:", round((time.time() - start) / 2 * mut_epochs, 3), "s")
print("best generation:", str(model1.best_gen) + ", best fit:", sp.fit_percent_str(model1.best_fit, 2, img_size))

x = np.arange(0, 2*epochs, 1)
fig1 = plt.Figure()
plt.plot(x, fit_arr1)
plt.show()

fig2 = plt.Figure()
plt.plot(x, fit_arr2)
plt.show()

# ___________________________ save images ___________________________________

im = Image.new('L', (img_size, img_size))
im.putdata(model1.best_img)
im.save('huygens1.png')

plate_print = 255 * model1.best_plate
platim = Image.new('L', (plate_size, plate_size))
platim.putdata(plate_print.flatten('C'))
platim.save('plate1.png')

# ___________________________ save images ___________________________________

im = Image.new('L', (img_size, img_size))
im.putdata(model2.best_img)
im.save('huygens2.png')

plate_print = 255 * model2.best_plate
platim = Image.new('L', (plate_size, plate_size))
platim.putdata(plate_print.flatten('C'))
platim.save('plate2.png')

# ein = Image.new('L', (image_size, image_size))
# ein.putdata(newarr)
# ein.save('zweistein.png')
