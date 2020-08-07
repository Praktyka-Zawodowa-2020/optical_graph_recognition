# PRAKTYKA ZAWODOWA 2020
# Optical Graph Recognition - Algorithm

This algorithm was created as part of the professional practice 2020 at the Gdańsk University of Technology.

## About

Written in python 3.7, it uses [OpenCV](https://docs.opencv.org/master/)

This algorithm works with the server whose repository you can see [here](https://github.com/Praktyka-Zawodowa-2020/optical_graph_recognition_server).

It can also be run from the command line.


This algorithm recognizes graphs from photos. His work is satisfactory, but for better effects, the photo sent to the algorithm should be well lit, and the contrast between the background and the graph should be high. The vertices in the graph should be ellipses. The algorithm can recognize a vertex from an unclosed ellipse, but it is recommended that they be closed. The shape of the ellipse should be similar to a circle. The algorithm tries to remove characters and noise from photos.

## How it's working

Its operation has been divided into 4 phases:
1. Preprocessing
2. Segmentation
3. Topology recognition
4. Postprocessing

### Preprocessing

In this phase, the input image undergoes the process of binarization and character removal. 

### Segmentation

In this phase, vertices are detected in the image after preprocessing and then filled. Once filled, the vertices are separated from the edges and other noise remaining in the image. The separated vertices are recognized and saved as a vertex list.

### Topology Recognition

In this phase, an attempt is made to detect the topology of the graph. The result of this phase is a adjacency lists.

### Postprocesing

In post-processing based on the vertex list, the recognized graph was saved to the * Graph6 * and * GraphML * formats

## Run from the command line

```
ATTENTION!!! Recognition of the file format is implemented on the server, 
so check carefully whether the path to the file indicates a photo in the * JPG * or * PNG * format
```

To run a script from the command line, type:

`<path_to_python>\python.exe <path_to main.py> [-p path_to_photo] [-b background]`

To see more information, please enter:

`<path_to_python>\python.exe <path_to main.py> -h`
 
Example when you are in the script folder:

`<path_to_python>\python.exe main.py -p <path_fo_file> -b AUTO`

Background mode
```
GRID_BG - Hand drawn on grid/lined piece of paper (grid/lined notebook etc.) \n
CLEAN_BG - Hand drawn on empty uniform color background \n
PRINTED - Printed (e.g. from paper, publication, book...)\n
AUTO - Mode is chosen automatically
```

## For programmers and project developers

To see the results of each phase in the main function for debugging, replace the flags in the "preprocess()", "segment()", "topology_recognition()" functions from False to True.

The problems that the developers of this software will face are:
1. If an edge is broken, the algorithm does not detect it.
2. Detection of intersecting edges.
3. Detection of edges that are not straight lines


## References

When creating the project, the authors were inspired by this [work](https://link.springer.com/content/pdf/10.1007%2F978-3-642-36763-2_47.pdf).

Some solutions have been taken and adapted from [yWorks](https://www.yworks.com/blog/projects-optical-graph-recognition).

#### Filip Chodziutko, Kacper Nowakowski 2020
