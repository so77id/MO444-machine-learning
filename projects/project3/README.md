# Commands

`make setup`: Download and unzip dataset

`make main`: Get the values of silhouette for specific kmeans with specific k, and number of components, also obtains the distributions of the clusters.
* `K_PCA`: Number of components of PCA
* `N_CLUSTERS`: number of cluster for k-means

`make process_docs`: process dataset and get the most used words in each document

`make get_new_names`: make graph with the distribution of labels in dataset

`make kmeans_search`: make search of k in kmeans using elbow and silhouette coefficient using this variables:
* `STEP`: step of k
* `INIT_K`: first value of k
* `END_K`: last value of k

`make kmeans_with_pca_search`:make search of k in kmeans with PCA using elbow and silhouette coefficient using this variables:
* `STEP`: step of k
* `INIT_K`: first value of k
* `END_K`: last value of k
* `K_PCA`: number of components of PCA

`make kpca_search`: make search of number of components of PCA using:
* `VARIANCE_THRESHOLD`: for search the first component that get this accumulated variance