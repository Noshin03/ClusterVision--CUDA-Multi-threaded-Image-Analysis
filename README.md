# ClusterVision: CUDA Multi-threaded Image Analysis
This project presents an innovative approach to improve image segmentation in computer vision. By integrating Principal Component Analysis (PCA) with K-Means clustering and CUDA parallelization, we significantly enhanced accuracies and processing speeds in image segmentation. PCA aids in efficient data handling and feature extraction, while K-Means clustering ensures precise data segmentation. CUDA parallelization leverages GPU computing for rapid execution. This method not only refines the accuracy of image segmentation but also addresses the challenges of real-time processing and large dataset management
## Getting started
Check cuda version. make share your cuda version is cuda_11.8
```
nvcc --version
```
install Eigen/Dense library
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
```
### Alternative; Google colab
```
https://colab.research.google.com/drive/10-OLSOj2MCg3LAly-aQxLJPygKebI2-L?usp=sharing
```
## Project structure
<p align=center>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/structure.jpg">
</p>

### PCA
* Step 1: Standardize the Data
* Step 2: Compute the Covariance Matrix
* Calculate Eigenvectors and Eigenvalues
* Sort Eigenvectors by Eigenvalues
* Choose the Top k Eigenvectors
* Project the Data onto Lower-Dimensional Space
#### Runtime result
after parallelize with CUDA, compare runtime with different size of data
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_time_compare.png">
</p>
compare runtime with different number of thread
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_THREAD_RUNTIME.png">
</p>
