import keras
import numpy as np
import os
import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image


def load_models():
    plane_prediction = keras.models.load_model("./plane_model_light_v2.h5")  # loading model for plane prediction
    xy = keras.models.load_model("./xy_model_light_v1.h5")  # these 3 lines load models to predict errors for each plane
    xz = keras.models.load_model("./xz_model_light_v1.h5")
    yz = keras.models.load_model("./yz_model_light_v1.h5")
    return plane_prediction, xy, xz, yz


def predict_errors(folder, file):  # makes a prediction for a VOID, PLF
    global models
    plane_labels = {
        0: "XY",
        1: "XZ",
        2: "YZ"
    }
    # preprocessing for plane classification (img_plane) and error prediction (img)
    img = cv2.imread(os.path.join(folder, file))
    img_plane = (cv2.cvtColor(cv2.resize(img, (500, 500)), cv2.COLOR_BGR2GRAY))
    img_plane = img_plane/255
    img_plane = img_plane.reshape(-1, img_plane.shape[0], img_plane.shape[1], 1)

    plane = plane_labels[np.argmax(models["Plane"].predict(img_plane), axis=1)[0]]  # predict the plane
    # resize the image depending on the plane, so that it doesn't crash if the plane is predicted wrong
    if plane == "XY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif plane == "XZ":
        img = cv2.cvtColor(cv2.resize(img, (512, 500)), cv2.COLOR_BGR2GRAY)
    elif plane == "YZ":
        img = cv2.cvtColor(cv2.resize(img, (500, 512)), cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float32)
    img = img/255
    img = img.reshape(-1, img.shape[0], img.shape[1], 1)

    model = models[plane]  # choose the correct model for a certain plane
    prediction = np.argmax(model.predict(img), axis=1)  # make the prediction

    # Write down the error into statistics:
    if plane == "YZ" or plane == "XZ":
        error_labels = {
            0: "No error",
            1: "A PLF"
        }
        if prediction[0] == 1:
            statistics["messages"].append(f"A PLF was found in {file}\n") 
            statistics["PLFs_found"] += 1

    elif plane == "XY":
        error_labels = {
            0: "No error",
            1: "A Void"
        }
        if prediction[0] == 1:
            statistics["messages"].append(f"A void was found in {file}\n") 
            statistics["voids_found"] += 1

    return prediction[0]


def one_by_one(folder):
    canvas.itemconfig(textbox_alert, text="Please wait for all the images to scan")
    global statistics
    for file in os.listdir(folder):  # takes images one-by-one and predicts them
        statistics["images_scanned"] += 1
        canvas.itemconfig(textbox_count, text=statistics["images_scanned"])
        predicted = predict_errors(folder, file)
        if predicted != 0:
            statistics["errors_found"] += 1
        canvas.itemconfig(textbox_errors, text=f"{statistics['errors_found']} ({round(statistics['errors_found'] / statistics['images_scanned'] * 100)}%)")
        canvas.update()  # updates the live-count
    canvas.itemconfig(textbox_alert, text="All images have been scanned,\n you can see the full statistics")
    canvas.update()


def choosing_folder(one):
    # creates a window to choose a folder
    win = Tk()
    # Define the geometry
    win.geometry("750x250")

    def select_file():
        folder = filedialog.askdirectory()
        Label(win, text=folder, font=13).pack()
        win.destroy()
        return one_by_one(folder)

    # Create a label and a Button to Open the dialog
    Label(win, text="Click the Button to Select a folder", font='Aerial 18 bold').pack(pady=20)
    button = ttk.Button(win, text="Select", command=select_file)
    button.pack(ipadx=5, pady=15)
    win.mainloop()


def show_statistics(one):
    # creates a window with full statistics described using the statistics dictionary
    win = Tk()
    win.geometry("750x250")
    scrollbar = Scrollbar(win)
    scrollbar.pack(side=RIGHT, fill=Y)
    textbox = Text(win)
    textbox.pack()
    textbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=textbox.yview)
    textbox.insert(END, f"Images scanned: {statistics['images_scanned']}\n")
    if statistics['images_scanned'] != 0:
        textbox.insert(END, f"Errors found: {statistics['errors_found']} ({round(statistics['errors_found'] / statistics['images_scanned'] * 100)}%  of images scanned)\n")
    else:
        textbox.insert(END, f"Errors found: 0\n")
    textbox.insert(END, f"Voids found: {statistics['voids_found']}\n")
    textbox.insert(END, f"PLFs found: {statistics['PLFs_found']}\n")
    textbox.insert(END, f"\nThe following are the errors and corresponding images:\n")
    textbox.tag_add("bold", "6.0", "6.54")
    textbox.tag_config("bold", background="green", foreground="white")
    textbox.update()
    for i in statistics["messages"]:
        textbox.insert(END, i)
    textbox.update()
    win.mainloop()


def gui():
    root = Tk()
    root.title = "The Program that still needs a name - Version -1 Alpha"
    root.geometry("1920x1080")

    # create a frame
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=YES)

    # defining images
    bg = PhotoImage(file="GUI/Background.png")
    importimage = ImageTk.PhotoImage(Image.open("GUI/import_button.png"))
    scannedimage = ImageTk.PhotoImage(Image.open("GUI/scanned_errors.png"))
    statisticsimage = ImageTk.PhotoImage(Image.open("GUI/statistics_middle.png"))
    textbox_bg_image = ImageTk.PhotoImage(Image.open("GUI/action_display.png"))

    # Create a canvas
    global canvas
    canvas = Canvas(frame, width=1920, height=1080)
    canvas.pack(fill="both", expand=YES)

    # set the background
    canvas.create_image(0, 0, image=bg, anchor="nw")

    # create buttons
    importbutton = canvas.create_image(1150, 150, image=importimage)
    canvas.create_image(300, 400, image=scannedimage)
    statisticsbutton = canvas.create_image(1150, 400, image=statisticsimage)
    canvas.create_image(400, 150, image=textbox_bg_image)

    # tag buttons to functions
    canvas.tag_bind(importbutton, "<Button-1>", choosing_folder)
    canvas.tag_bind(statisticsbutton, "<Button-1>", show_statistics)

    # create updatable texts
    global textbox_count
    global textbox_errors
    global textbox_alert
    textbox_count = canvas.create_text(410, 368, text="0", fill="white", font="MyriadPro 28")
    textbox_errors = canvas.create_text(410, 430, text="0", fill="white", font="MyriadPro 28")
    textbox_alert = canvas.create_text(400, 150, text="Please choose images to scan", fill="#fff59e", font="MyriadPro 28")

    root.mainloop()


# main part of the code
plane_classification, xy_model, xz_model, yz_model = load_models()  # loading the models
statistics = {  # create statistics dictionary
    "images_scanned": 0,
    "errors_found": 0,
    "voids_found": 0,
    "PLFs_found": 0,
    "messages": []
}
models = {  # create model dictionary
    "Plane": plane_classification,
    "XY": xy_model,
    "XZ": xz_model,
    "YZ": yz_model
}
gui()  # starts up the GUI
