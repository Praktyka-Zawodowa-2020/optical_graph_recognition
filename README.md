# PRAKTYKA ZAWODOWA 2020

## Optical Graph Recognition - Algorithm

This algorithm works with the server whose repository you can see [here](https://github.com/Praktyka-Zawodowa-2020/optical_graph_recognition_server).
It can also be run from the command line.

This algorithm was created as part of the professional practice 2020 at the Gdańsk University of Technology
Its operation has been divided into 4 phases:
1. Preprocessing
2. Segmentation
3. Topology recognition
4. Postprocessing

## Preprocesing

In this phase, the input image undergoes the process of binarization and character removal. 

### Segmentation

In this phase, vertices are detected in the image after preprocessing and then filled. Once filled, the vertices are separated from the edges and other noise remaining in the image. The separated vertices are recognized and saved as a vertex list.

### Topology Recognition

In this phase, an attempt is made to detect the topology of the graph. The result of this phase is a vertex list together with neighborhood lists.

### Postprocesing

In post-processing based on the vertex list, the recognized graph was saved to the * Graph6 * and * GraphML * formats

## Run from the command line

To run a script from the command line, type:
'<path_to_python>\python.exe <path_to main.py> [-p path_to_photo] [-b background]'

To see more information, please enter:
'<path_to_python>\python.exe <path_to main.py> -h'
 
Example when you are in the script folder:
'<path_to_python>\python.exe main.py -p <path_fo_file> -b AUTO'

Background mode
'GRID_BG - Hand drawn on grid/lined piece of paper (grid/lined notebook etc.) 
CLEAN_BG - Hand drawn on empty uniform color background 
PRINTED - Printed (e.g. from paper, publication, book...)
AUTO - Mode is chosen automatically'


### For programmers and project developers
If you want to see the results of particular phases in the main function, replace the flags in the "preprocess ()", "segment ()", "recognize_topology ()" functions from False to True

The problems that the developers of this software will face are:
1. Detect edges that are broken.
2. Detection of intersecting edges.
3. Detection of edges that are not straight lines

#### Filip Chodziutko, Kacper Nowakowski 2020
