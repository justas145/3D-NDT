THIS IS A DESCRIPTION OF ALL THE FILES IN THE FOLDER.

1) {plane}_model_light_{version}.h5: the CNN models that are used for detecting the errors in the corresponding planes.

2) plane_model_light_v2.h5: the CNN model that is used for identifying the plane of the image.

3) CV_Capstone.py: The program with GUI, which predicts the errors (Voids and PLFs) for all the images that are uploaded. When running the program, click on "Import/Select Folder" and then choose a folder were the OCT scans are stored. Then the program will start analyzing the images, the progress can be seen on the main screen with the count of images scanned, and errors found (the number in the parenthesis is the percentage of erros in the scanned images). When all the images would be processed the you will be notified on the main window. After that you can press the "Statistics" button to see the full statistics.

4) GUI folder: a folder that contains all the images used in the GUI of CV_Capstone.py

5) stitch.py: Standalone python file used to visualize the scans in 3D. Uses pyvista. Note that paraview can also be used.

6) run.py: Boilerplate code used to run the classification algorithms in different axes. Most important parameters can be set between lines 13 and 33. The image processing algorithms are unique to each plane and hence in a file labeled “{plane}.py”, then imported into run.py. Changing modes is probably the most important setting, this switches between the different ways to process images. For example, mode 0 selects a few samples of images and processes them. They idea is that the algorithms can be developed with just a few images. Then a random 10 images can be sampled with this algorithm in mode 3. Different modes have found different uses throughout the project

7) gc_v2.py: Used to visualize the performance of a CNN. Importantly, it requires a CNN model (h5 filetype) and at least one image to run the CNN on. It then selects the layer named “just_do_it” to perform the Grad-CAM software on.

8) scatter_plot.py: Used to create a scatterplot from "run.py" mode 5 operation. Note that mode 5 will create a csv file that needs to be manually edited to identify at least one defect. Then, scatter.py will automatically change its color and one is able to hover above point on the scatter plot and manually investigate if the image has a defect.

9) plane_identification_light_model.ipynb: a notebook, that loads in the images, preprocess them, generates learning curves, confusion matrix and trains a CNN model to identify planes. First the dataset is generated with the function "load_images".
If you want to load your own images, an example of inputed XY text file line should look like this: Processed_Images_XY\Cube2CornerXY252.jpg:0. For YZ: Cube1CornerYZ9.jpg:0.
For XZ: Cube1CornerXZ9.jpg:0. The number after ":" doesn't matter, the labels are generated based on the axis name in the file name. Also put your XY images in "Processed_Images_XY" folder.
YZ images in "Processed_Images_YZ". XZ images in "Processed_Images_XZ"

10) xy_light_and_heavy_model.ipynb: a notebook that that loads in the images, preprocess them, generates learning curves, confusion matrix and trains and saves light and heavy CNN models.Image loading is same as in plane_identification_light_model.ipynb, except
only load text file with XY images.

11) xz_light_and_heavy_model.ipynb: Creates augmented images, loads and preprocess images, created learning curve plots, confusion matrix and trains light and heavy CNN models. To create augmented images, need to provide an image folder.
It will create a folder called output in the folder with the images that are used for augmentation.
To load the images either provide again a text file(example:Cube1CornerXZ9.jpg:0) and a folder name with the images in it for the function "load_images_xz". Or you can provide a list with
images names and a folder name with the images. Also, if you do provide a list, set list=True, and if those images in that folder are with error, set error=True.

12) plane_identification_svm_model.ipynb: a notebook, that loads in the images, preprocess them and trains a SVM model to identify planes. To load the images it's same process as with plane_identification_light_model.ipynb.

13) yz_light_and_heavy_model.ipynb: everything is the same as xz_light_and_heavy_model.ipynb, just use yz images.

INSTALLATION:

In the IDE where you run python go to the terminal and run "pip install {library}" for the following libraries in order for all the files to work properly.

libraries used:
1) keras==2.7.0
2) numpy==1.20.2
3) opencv-python==4.5.4.60
4) tk==8.6.10
5) tensorflow==2.7.0
6) pillow==8.4.0
7) pyvista==0.33.2
8) matplotlib==3.5.1
9) augmentor==0.2.9
10) scikit-learn==0.24.2
11) scipy==1.7.3





