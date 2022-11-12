from tkinter import Tk, PhotoImage, mainloop, Canvas
from idx_format_converter import IDX

IMAGE_SIZE = 28
SCALE = 1
WINDOW_SIZE = IMAGE_SIZE * SCALE


def draw_image(filename, image_number):
    window = Tk()
    image = PhotoImage(width=WINDOW_SIZE, height=WINDOW_SIZE)

    canvas = Canvas(window, width=WINDOW_SIZE, height=WINDOW_SIZE, bg='#000000')
    canvas.pack()
    canvas.create_image((WINDOW_SIZE / 2, WINDOW_SIZE / 2), image=image, state="normal")

    paint(image, window, filename, image_number)

    print("Close the image window to exit the program")
    mainloop()


def draw_all(filename):
    window = Tk()
    image = PhotoImage(width=WINDOW_SIZE, height=WINDOW_SIZE)

    canvas = Canvas(window, width=WINDOW_SIZE, height=WINDOW_SIZE, bg='#000000')
    canvas.pack()
    canvas.create_image((WINDOW_SIZE / 2, WINDOW_SIZE / 2), image=image, state="normal")

    data = IDX(filename)
    for i in range(data.get_dimensions()[0]):
        paint(image, window, filename, i)
        window.after(50)

    print("Close the image window to exit the program")
    mainloop()


def draw_batch(filename, start, count):
    # TODO: write draw_batch()
    pass


def paint(image, window, filename, image_number):
    data = IDX(filename)
    for row in range(WINDOW_SIZE):
        for col in range(WINDOW_SIZE):
            x = col // SCALE
            y = row // SCALE
            pixel_color = data.get_val((image_number, y, x)).hex()
            color = "#" + pixel_color + pixel_color + pixel_color
            image.put(color, (col, row))
    window.update()
