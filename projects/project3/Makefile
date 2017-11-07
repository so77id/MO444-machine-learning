# rcnn-fer Docker Makefile
PROGRAM="RCNN-FER"

CPU_REGISTRY_URL=so77id
CPU_VERSION=latest
CPU_DOCKER_IMAGE=tensorflow-opencv-cpu-py3


#COMMANDS
PYTHON_COMMAND=python3
WGET_COMMNAND=wget
UNZIP_COMMAND=unzip
MV_COMMAND=mv
RM_COMMAND=rm

# FILES
DATASET_ZIP_FILE=documents.zip
DATASET_ZIP_FOLDER=documents

#SRC FILES
MAIN_FILE=main.py
KMEANS_SEARCH_FILE=k_search_kmeans.py
KPCA_SEARCH_FILE=k_search_pca.py

#FOLDERS
DATABASE_FOLDER=./dataset
IMG_FOLDER=./imgs

#URLs
DATASET_URL=https://www.dropbox.com/s/fjtbwf7f5p9f3lx/documents.zip


#INPUTS
INPUT_FILE=$(DATABASE_FOLDER)/data.csv
IDS_FILE=$(DATABASE_FOLDER)/ids
METHOD_NAME=baseline
# METHOD_NAME=dbscan
# METHOD_NAME=tsne

#OUTPUTS
OUTPUT_GRAPH_DISTORITION=$(IMG_FOLDER)/distortion_graph.png
OUTPUT_GRAPH_SILHOUETTE=$(IMG_FOLDER)/silhouette_graph.png


OUTPUT_GRAPH_PCA1=$(IMG_FOLDER)/pca_variance_graph.png
OUTPUT_GRAPH_PCA2=$(IMG_FOLDER)/pca_accumulated_variance_graph.png


#MAGIK NUMBERS
STEP=25
INIT_K=25
END_K=500

VARIANCE_THRESHOLD=0.80

#90%
# K_PCA=1274
# 80%
K_PCA=849
N_CLUSTERS=190

all: main

setup s:
	@echo "[Setup] Setup enviroment"
	@$(WGET_COMMNAND) $(DATASET_URL)
	@$(UNZIP_COMMAND) $(DATASET_ZIP_FILE)
	@$(RM_COMMAND) $(DATASET_ZIP_FILE)
	@$(MV_COMMAND) $(DATASET_ZIP_FOLDER) $(DATABASE_FOLDER)


main m:
	@echo "[Main] Executing"
	@# $(PYTHON_COMMAND) $(MAIN_FILE) -i $(INPUT_FILE) -ds $(DESCRIPTOR_SIZE) -nd $(N_DESCRIPTORS)
	@$(PYTHON_COMMAND) $(MAIN_FILE) -i $(INPUT_FILE) -n_components $(K_PCA) -k $(N_CLUSTERS)

kmeans_search kms:
	@echo "[Search kmeans] Searching k"
	@$(PYTHON_COMMAND) $(KMEANS_SEARCH_FILE) -i $(INPUT_FILE) -s $(STEP) -ik $(INIT_K) -fk $(END_K) -od $(OUTPUT_GRAPH_DISTORITION) -os $(OUTPUT_GRAPH_SILHOUETTE)
kmeans_with_pca_search kmspca:
	@echo "[Search kmeans] Searching k with pca"
	@$(PYTHON_COMMAND) $(KMEANS_SEARCH_FILE) -i $(INPUT_FILE) -s $(STEP) -ik $(INIT_K) -fk $(END_K) -od $(OUTPUT_GRAPH_DISTORITION) -os $(OUTPUT_GRAPH_SILHOUETTE) -pca -k_pca $(K_PCA)
kpca_search kps:
	@echo "[Search pca] Searching k"
	@$(PYTHON_COMMAND) $(KPCA_SEARCH_FILE) -i $(INPUT_FILE) -o1 $(OUTPUT_GRAPH_PCA1) -o2 $(OUTPUT_GRAPH_PCA2) -t $(VARIANCE_THRESHOLD)