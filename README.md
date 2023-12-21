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
##### **after parallelize with CUDA, compare runtime with different size of data**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_time_compare.png">
</p>
With increasing data sizes, both CPU and GPU runtimes rise, initially with the GPU lagging behind for smaller data. The non-linear increase in runtime suggests escalating processing complexity. Despite GPUs offering advantages for larger datasets, efficient algorithm optimization is crucial, given the close runtime proximity of GPU and CPU, influenced by the intensive eigen-decomposition process in PCA operations.

##### **compare runtime with different number of thread**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/PCA_THREAD_RUNTIME.png">
</p>
The GPU runtime exhibits a sharp decrease with an increasing number of threads, particularly up to 16 threads, where the reduction is most pronounced, but beyond 64 threads, the rate of decrease becomes less significant, suggesting a diminishing return on runtime reduction due to parallelization overhead and indicating an optimal range of thread utilization for significant performance gains.

### K_MEAN
* Step 1: Choose the Number of Clusters (K)
* Step 2: Initialize Cluster Centers
* Step 3: Assign Data Points to Nearest Cluster
* Step 4: Update Cluster Centers
* Step 5: Repeat Steps 3 and 4
#### Runtime result
##### **after parallelize with CUDA, compare runtime with different size of data**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/K_MEAN_runtime_compare.png">
</p>
The CPU runtime shows a linear relationship, approximately Time = Number of Pixels / 595, for large data sizes (starting from 128128), while the GPU runtime follows Time = Number of Pixels / 6400. In the K-mean part, parallelization enhances computation speed by around 11 times, showcasing its effectiveness. Notably, for small data sizes (3232 and 64*64), multiplying image size by four only improves execution speed by 1.265 times due to sequential overheads in setting up parallelized computation processes.

##### **compare runtime with different number of thread**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/k_mean_THREAD_runtime.png">
</p>

### Image output Result

#### single object

**origional**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4.png" width="400" height="300"/>
</p>

**result**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4_out.png" width="400" height="300"/>
</p>

#### two objects

**origional**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4.png" width="400" height="300"/>
</p>

**result**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test4_out.png" width="400" height="300"/>
</p>
