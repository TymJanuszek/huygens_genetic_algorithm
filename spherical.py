import numpy as np
from numba import njit
from PIL import Image
from copy import deepcopy

R = 1  # plate-screen distance
k = np.pi / 12

pixel_size = 0.4

plate_div = 1


class HuygensModel:
    def __init__(self, goal_img, img_size, plate_size, name):
        empty_img = np.zeros(img_size ** 2, dtype=np.uint8)
        self.best_img = empty_img
        self.best_gen = 0
        self.best_plate = random_plate(0, 10)

        self.best_fit = np.sum(np.abs(empty_img - goal_img))
        self.fit_arr = []

        self.img_size = img_size
        self.plate_size = plate_size
        self.goal_img = goal_img

        self.model_name = name

    def one_point_algorithm(self, epochs):
        print("_______________________________________")
        print(self.model_name + " one point mutation")
        print("_______________________________________")
        plate = random_plate(100)
        self.fit_arr = np.zeros(epochs)

        for gen in range(epochs):
            # for i in range(int(img_size ** 2 / epochs /100)):
            #     plate = puncture(plate)

            plate = puncture(plate, self.plate_size)

            new_img_2d = np.zeros(shape=(self.img_size, self.img_size))
            new_img_2d = simulate(new_img_2d, plate)

            new_img = 255 * new_img_2d / np.max(new_img_2d)
            new_img = np.round(np.abs(new_img.flatten()))

            new_fit = np.sum(np.abs(new_img - self.goal_img))
            self.fit_arr[gen] = fit_percent(new_fit, self.img_size)

            print(gen, "generation fitness: ", fit_percent_str(new_fit, 4, self.img_size))

            if new_fit < self.best_fit:
                self.best_img = new_img
                print(" new best!")
                self.best_plate = np.copy(plate)
                self.best_fit = new_fit
                self.best_gen = gen

                plate_print = 255 * self.best_plate
                platim = Image.new('L', (self.plate_size, self.plate_size))
                platim.putdata(plate_print.flatten('C'))
                platim.save('platevolution/bestplate' + self.model_name + str(gen) + '.png')

                im = Image.new('L', (self.img_size, self.img_size))
                im.putdata(self.best_img)
                im.save('platevolution/huygens' + self.model_name + str(gen) + '.png')

            plate = np.copy(self.best_plate)

    def cross(self, second_model):
        random_rows = np.random.randint(self.img_size, size=int(self.img_size / 2))
        for index in random_rows:
            second = second_model.best_plate[index]

            self.best_plate[index] = second

        first_img_2d = np.zeros(shape=(self.img_size, self.img_size))
        first_img_2d = simulate(first_img_2d, self.best_plate)
        first_img = 255 * first_img_2d / np.max(first_img_2d)
        self.best_img = np.round(np.abs(first_img.flatten()))

        self.best_fit = np.sum(np.abs(self.best_img - self.goal_img))

        # plate_print = 255 * self.best_plate`
        # platim = Image.new('L', (self.plate_size, self.plate_size))
        # platim.putdata(plate_print.flatten('C'))
        # platim.save('platevolution/bestplate' + self.file_flag + str(gen) + '.png')
        #
        # im = Image.new('L', (self.img_size, self.img_size))
        # im.putdata(self.best_img)
        # im.save('platevolution/huygens' + self.file_flag + str(gen) + '.png')

        plate = np.copy(self.best_plate)


def cross_and_evaluate(first_model, second_model, cross_number=0):
    print("_______________________________________")
    print("cross " + first_model.model_name + " & " + second_model.model_name)
    print("_______________________________________")
    prefirst = deepcopy(first_model)
    presecond = deepcopy(second_model)

    first_model.cross(second_model)
    second_model.cross(prefirst)

    first_model.model_name = "_first_cross_" + str(cross_number)
    second_model.model_name = "_second_cross_" + str(cross_number)

    models = [prefirst, presecond, first_model, second_model]

    best = deepcopy(first_model)
    second_best = deepcopy(first_model)
    for model in models:
        print(model.model_name + " fitness: ", fit_percent_str(model.best_fit, 4, model.img_size))
        if model.best_fit < best.best_fit:
            second_best = best
            best = model
        elif model.best_fit < second_best.best_fit and model.best_fit:
            second_best = model

    best.model_name = "best_cross_" + str(cross_number) + "_"
    second_model.model_name = "2nd_best_cross_" + str(cross_number) + "_"

    print("results:")
    print(" best fitness: ", fit_percent_str(best.best_fit, 4, best.img_size))
    print(" 2nd best fitness: ", fit_percent_str(second_best.best_fit, 4, second_best.img_size))

    return best, second_best


def fit_percent(fit, img_size):
    return 100 - 100 * fit / (img_size * img_size * 255)


def fit_percent_str(fit, rnd, img_size):
    x = 100 - 100 * fit / (img_size * img_size * 255)
    return (str(round(x, rnd))) + "%"


def fitness_func(spec, goal):
    return np.sum(abs(spec - goal))


def load_image(path):
    img = Image.open(path)
    arr = np.asarray(img).flatten()
    return arr


@njit
def random_plate(plate_size, prob=0, out_of=100):
    plate = np.zeros(shape=(plate_size, plate_size))

    for j in (range(plate_size)):
        for i in (range(plate_size)):
            plate[j][i] = int(np.random.randint(0, out_of) < prob)

    return plate


@njit
def puncture(plate, plate_size):
    i, j = int(np.random.randint(0, plate_size)), int(np.random.randint(0, plate_size))
    plate[i][j] = 1
    return plate


@njit
def mid_dist(x, y, img_size):
    return pixel_size * np.sqrt((x - img_size / 2) ** 2 + (y - img_size / 2) ** 2)


@njit
def spherical(x, y, xoffset, yoffset, plate_size, img_size):
    A = 500 / np.sqrt(2 * (img_size / 2) ** 2 + R ** 2)
    r = mid_dist(x - xoffset + plate_div * plate_size / 2, y - yoffset + plate_div * plate_size / 2,
                 img_size) ** 2 + R ** 2
    return A / r * np.sin(k * r)


@njit
def simulate(image, plate):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for i in (range(plate.shape[0])):
                for j in (range(plate.shape[1])):
                    image[y][x] += plate[j][i] * spherical(x, y, plate_div * i, plate_div * j, plate.shape[0],
                                                           image.shape[0])

    return image
