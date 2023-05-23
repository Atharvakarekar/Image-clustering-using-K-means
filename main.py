# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
#
# # Set the path to the directory containing your landmark images
# image_directory = r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0'
#
# # Set the number of clusters
# num_clusters = 5
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize image to a common size if needed
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# # Load and preprocess images
# images = []
# for filename in os.listdir(image_directory):
#     if filename.endswith(('.jpg', '.png')):
#         image_path = os.path.join(image_directory, filename)
#         img = preprocess_image(image_path)
#         images.append(img)
#
# # Convert images to feature vectors
# feature_vectors = np.reshape(images, (len(images), -1))
#
# # Perform K-means clustering
# kmeans = KMeans(n_clusters=num_clusters)
# kmeans.fit(feature_vectors)
#
# # Get the cluster labels for each image
# cluster_labels = kmeans.labels_
#
# # Print the cluster labels for each image
# for i, filename in enumerate(os.listdir(image_directory)):
#     if filename.endswith(('.jpg', '.png')):
#         print(f'{filename}: Cluster {cluster_labels[i]}')

# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# import streamlit as st
#
# # Set the path to the directory containing your landmark images
# image_directory = r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0'
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize image to a common size if needed
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# def main():
#     st.title("Image Clustering")
#
#     # Sidebar for user inputs
#     st.sidebar.title("Parameters")
#     num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
#
#     # Load and preprocess images
#     images = []
#     for filename in os.listdir(image_directory):
#         if filename.endswith(('.jpg', '.png')):
#             image_path = os.path.join(image_directory, filename)
#             img = preprocess_image(image_path)
#             images.append(img)
#
#     # Convert images to feature vectors
#     feature_vectors = np.reshape(images, (len(images), -1))
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(feature_vectors)
#
#     # Get the cluster labels for each image
#     cluster_labels = kmeans.labels_
#
#     # Print the cluster labels for each image
#     for i, filename in enumerate(os.listdir(image_directory)):
#         if filename.endswith(('.jpg', '.png')):
#             # st.write(f'{filename}: Cluster {cluster_labels[i]}')
#             st.image(img, caption=f'{filename}: Cluster {cluster_labels[i]}')
#
# if __name__ == "__main__":
#     main()

# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# import streamlit as st
#
# # Set the path to the directory containing your landmark images
# image_directory = r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0'
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img.thumbnail((150, 150))  # Reduce image size to fit in the frame
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# def main():
#     st.title("Image Clustering")
#
#     # Sidebar for user inputs
#     st.sidebar.title("Parameters")
#     num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
#
#     # Load and preprocess images
#     images = []
#     for filename in os.listdir(image_directory):
#         if filename.endswith(('.jpg', '.png')):
#             image_path = os.path.join(image_directory, filename)
#             img = preprocess_image(image_path)
#             images.append(img)
#
#     # Convert images to feature vectors
#     feature_vectors = np.array(images)  # Convert images list to numpy array
#
#     # Reshape feature vectors if necessary
#     if len(feature_vectors.shape) > 2:
#         feature_vectors = feature_vectors.reshape(len(images), -1)
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(feature_vectors)
#
#     # Get the cluster labels for each image
#     cluster_labels = kmeans.labels_
#
#     # Display the images along with their names
#     images_per_row = 5
#     num_images = len(images)
#     num_rows = (num_images + images_per_row - 1) // images_per_row
#     for i, filename in enumerate(os.listdir(image_directory)):
#         if filename.endswith(('.jpg', '.png')):
#             image_path = os.path.join(image_directory, filename)
#             img = Image.open(image_path)
#             img.thumbnail((150, 150))  # Reduce image size to fit in the frame
#             st.image(img, caption=f'{filename}: Cluster {cluster_labels[i]}',
#                      width=150, use_column_width=True)
#
# if __name__ == "__main__":
#     main()

# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# import streamlit as st
#
# # Set the paths to the directories containing your landmark images
# image_directories = [
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\1',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\2',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\3',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\4'
# ]
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize image to a common size if needed
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# def main():
#     st.title("Image Clustering")
#
#     # Sidebar for user inputs
#     st.sidebar.title("Parameters")
#     num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
#
#     # Load and preprocess images
#     images = []
#     for image_directory in image_directories:
#         for filename in os.listdir(image_directory):
#             if filename.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(image_directory, filename)
#                 img = preprocess_image(image_path)
#                 images.append(img)
#
#     # Convert images to feature vectors
#     feature_vectors = np.reshape(images, (len(images), -1))
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(feature_vectors)
#
#     # Get the cluster labels for each image
#     cluster_labels = kmeans.labels_
#
#     # Display the images along with their names and cluster labels
#     for image_directory in image_directories:
#         for i, filename in enumerate(os.listdir(image_directory)):
#             if filename.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(image_directory, filename)
#                 img = Image.open(image_path)
#                 st.image(img, caption=f'{filename}: Cluster {cluster_labels[i]}')
#
# if __name__ == "__main__":
#     main()

# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# import streamlit as st
#
# # Set the paths to the directories containing your landmark images
# image_directories = [
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\1',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\2',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\3',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\4'
# ]
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize image to a common size if needed
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# def main():
#     st.title("Image Clustering")
#
#     # Sidebar for user inputs
#     st.sidebar.title("Parameters")
#     num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
#
#     # Load and preprocess images
#     images = []
#     for image_directory in image_directories:
#         for filename in os.listdir(image_directory):
#             if filename.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(image_directory, filename)
#                 img = preprocess_image(image_path)
#                 images.append(img)
#
#     # Convert images to feature vectors
#     feature_vectors = np.reshape(images, (len(images), -1))
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(feature_vectors)
#
#     # Get the cluster labels for each image
#     cluster_labels = kmeans.labels_
#
#     # Display the images along with their names and cluster labels
#     columns = st.beta_columns(5)  # Adjust the number of columns as per your requirement
#     for image_directory in image_directories:
#         for i, filename in enumerate(os.listdir(image_directory)):
#             if filename.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(image_directory, filename)
#                 img = Image.open(image_path)
#                 with columns[i % 5]:
#                     st.image(img, caption=f'{filename}: Cluster {cluster_labels[i]}')
#
# if __name__ == "__main__":
#     main()

#
# import os
# from PIL import Image
# import numpy as np
# from sklearn.cluster import KMeans
# import streamlit as st
#
# # Set the paths to the directories containing your landmark images
# image_directories = [
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\1',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\2',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\3',
#     r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\4'
# ]
#
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize image to a common size if needed
#     img = img.convert("RGB")  # Convert image to RGB mode
#     return np.array(img)
#
# def main():
#     st.title("Image Clustering")
#
#     # Sidebar for user inputs
#     st.sidebar.title("Parameters")
#     num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
#
#     # Load and preprocess images
#     images = []
#     image_paths = []
#     for image_directory in image_directories:
#         for filename in os.listdir(image_directory):
#             if filename.endswith(('.jpg', '.png')):
#                 image_path = os.path.join(image_directory, filename)
#                 img = preprocess_image(image_path)
#                 images.append(img)
#                 image_paths.append(image_path)
#
#     # Convert images to feature vectors
#     feature_vectors = np.reshape(images, (len(images), -1))
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(feature_vectors)
#
#     # Get the cluster labels for each image
#     cluster_labels = kmeans.labels_
#
#     # Organize images by cluster
#     clusters = {}
#     for i, image_path in enumerate(image_paths):
#         cluster = cluster_labels[i]
#         if cluster not in clusters:
#             clusters[cluster] = []
#         clusters[cluster].append(image_path)
#
#     # Display the images along with their names and cluster labels
#     for cluster in clusters:
#         st.write(f"Cluster {cluster}")
#         columns = st.columns(5)  # Adjust the number of columns as per your requirement
#         for image_path in clusters[cluster]:
#             img = Image.open(image_path)
#             with columns[len(columns) % 5]:
#                 st.image(img, caption=os.path.basename(image_path))
#
# if __name__ == "__main__":
#     main()



import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

# Set the paths to the directories containing your landmark images
image_directories = [
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\0',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\1',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\2',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\3',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\4',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\5',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\6',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\7',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\8',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\9',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\a',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\b',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\c',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\d',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\e',
    r'C:\Users\Pratima\Downloads\image clustering\images_000\0\0\f',
]

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to a common size if needed
    img = img.convert("RGB")  # Convert image to RGB mode
    return np.array(img)

def main():
    st.title("Image Clustering")

    # Sidebar for user inputs
    st.sidebar.title("Parameters")
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

    # Load and preprocess images
    images = []
    image_paths = []
    for image_directory in image_directories:
        for filename in os.listdir(image_directory):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(image_directory, filename)
                img = preprocess_image(image_path)
                images.append(img)
                image_paths.append(image_path)

    # Convert images to feature vectors
    feature_vectors = np.reshape(images, (len(images), -1))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(feature_vectors)

    # Get the cluster labels for each image
    cluster_labels = kmeans.labels_

    # Organize images by cluster
    clusters = {}
    for i, image_path in enumerate(image_paths):
        cluster = cluster_labels[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(image_path)

    # Display the images along with their names and cluster labels
    for cluster in clusters:
        st.write(f"Cluster {cluster}")
        row_images = []
        for image_path in clusters[cluster]:
            img = Image.open(image_path)
            row_images.append(img)

        st.image(row_images, caption=[os.path.basename(path) for path in clusters[cluster]], width=150)

if __name__ == "__main__":
    main()
