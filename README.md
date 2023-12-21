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
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test5.jpg" width="250" height="300"/>
</p>

**result**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test5_out.png" width="250" height="300"/>
</p>

#### Multiple objects

**origional**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/sourceCode/Test3.png" width="250" height="300"/>
</p>

**result**
<p align=left>
    <img src="https://github.com/Noshin03/ClusterVision--CUDA-Multi-threaded-Image-Analysis/blob/main/ImageResources/test6%20result.png" width="250" height="300"/>
</p>

The algorithm doesn’t work well with low color contrast objects due to the random and limited optimization of k-means.

K-means clustering, while widely used, often encounters challenges in consistently delivering the desired results, primarily due to its sensitivity to initial conditions and data anomalies. The initial placement of centroids is crucial, as random selection can lead to varying clustering outcomes and potentially cause the algorithm to converge to a local minimum rather than the optimal global solution. Furthermore, K-means is particularly sensitive to outliers, which can skew centroid positions and result in unexpected clustering. Another critical factor is the choice of the 'k' value, the number of clusters; an inappropriate selection here can lead to suboptimal clustering, underscoring the importance of accurately determining 'k' for effective clustering.
In terms of future improvements, our team is considering transitioning from K-means to the Gaussian Mixture Model (GMM) for more precise image clustering. GMM offers a more sophisticated approach as it does not assume clusters to be spherical and can adapt to the actual distribution of data. This flexibility allows for more accurate clustering, particularly beneficial in complex image segmentation tasks where the data distribution is not uniform.
However, the parallelization of the Gaussian Mixture Model poses significant challenges and demands extensive development time. GMM's complexity, primarily due to its reliance on Expectation-Maximization (EM) for parameter estimation, makes it more computationally intensive and harder to parallelize efficiently compared to simpler algorithms like K-means. The EM algorithm in GMM involves iterative calculations that are highly dependent on previous steps, complicating the parallelization process.
Despite these challenges, the potential gains in clustering accuracy and segmentation quality make GMM a promising avenue for future research and development. Our team plans to invest time in developing strategies to parallelize GMM effectively, aiming to harness the power of GPU computing to make this sophisticated model more viable for large-scale and real-time applications in image analysis. This would involve overcoming significant technical hurdles but could ultimately lead to more versatile and accurate clustering methods in the field of computer vision.
For the next step in our project, our team plans to integrate data prefetching into both the PCA and K-means implementations. This advancement aims to enhance the performance of these algorithms by optimizing memory access patterns and reducing latency in data retrieval.
Incorporating data prefetching into K-Mean will target the acceleration of key steps such as finding labels and recalculating the centers. Given k-mean's reliance on large matrix operations, prefetching can significantly speed up these processes by ensuring that the necessary data is readily available in the GPU's cache, minimizing delays caused by data fetching from slower memory sources.


More details please refer to the report.pdf file

