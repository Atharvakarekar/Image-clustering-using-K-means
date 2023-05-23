# Image clustering using K-means

The given code performs image clustering using the K-means algorithm. Here's a step-by-step description of the code:

The code starts by importing the necessary libraries: os for file operations, PIL (Python Imaging Library) for image processing, numpy for numerical operations, sklearn.cluster.KMeans for performing K-means clustering, and streamlit for creating a user interface.

The code defines the path to the directory containing the landmark images using the image_directory variable.

The code defines a function preprocess_image that takes an image path as input and performs preprocessing on the image. The preprocessing includes opening the image, resizing it to a common size (224x224 pixels), and converting it to RGB mode. The function returns the image as a numpy array.

The code defines the main function, which is the entry point of the program. It sets the title of the Streamlit application to "Image Clustering".

The code adds a sidebar to the Streamlit application where the user can specify the number of clusters using a slider.

The code loads and preprocesses the images from the specified image_directory. It iterates over the files in the directory and checks if the file has a ".jpg" or ".png" extension. If it does, it constructs the image path and calls the preprocess_image function to preprocess the image. The preprocessed images are stored in a list called images.

The code converts the list of images to feature vectors by reshaping them into a 2D array. Each image is flattened into a 1D array.

The code performs K-means clustering using the KMeans class from scikit-learn. The number of clusters is set to the value chosen by the user in the sidebar.

The code assigns cluster labels to each image by calling the labels_ attribute of the KMeans object.

The code displays the images along with their names and cluster labels. It iterates over the files in the image_directory, checks if the file has a ".jpg" or ".png" extension, opens the image, and displays it using the st.image function from Streamlit. The caption of each image includes the filename and the corresponding cluster label.

The if __name__ == "__main__": condition ensures that the main function is only executed when the script is run directly, not when it is imported as a module.

This code can be run as a Streamlit application to interactively explore the image clustering results. The user can adjust the number of clusters using the sidebar slider, and the images will be displayed with their cluster labels. The code uses the st.image function to display the images in a grid layout, and the number of columns can be adjusted as needed.
 
![image](https://github.com/Atharvakarekar/Image-clustering-using-K-means/assets/91048746/06e487e7-a9ad-4612-bd51-fa9c46f90e50)
![image](https://github.com/Atharvakarekar/Image-clustering-using-K-means/assets/91048746/90c7da9e-053b-4b1f-bced-8cd8c98a43f5)
