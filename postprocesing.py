from Vertex import Vertex
from typing import List
from shared import Color


def graph6_format(vertex: List[Vertex], save_path: str):
    """
    Saves the graph in .graph6 format

    :param save_path: Folder path with the file name. No extension
    :param vertex: Lists of vertex
    :return:
    """

    save_path = save_path + '.g6'
    f = open(save_path, "wb")
    size = len(vertex)
    adjacency_matrix = [[0 for x in range(size)] for y in range(size)]
    adjacency_list = []
    for i in range(0, size):
        vertex[i].id = i

    for V in vertex:
        for W in V.neighbour_list:
            adjacency_matrix[V.id][W] = 1

    for i in range(0, size):
        for j in range(0, i):
            adjacency_list.append(adjacency_matrix[j][i])

    # align on the right with 0 so that the length is a multiple of 6.
    if len(adjacency_list) % 6 != 0:
        while len(adjacency_list) % 6 != 0:
            adjacency_list.append(0)

    if size < 63:
        f.write((size + 63).to_bytes(1, byteorder='big'))
    elif 63 <= size <= 258047:
        f.write((126).to_bytes(1, byteorder='big'))
    elif 258047 <= size <= 68719476735:
        f.write((126).to_bytes(1, byteorder='big'))
        f.write((126).to_bytes(1, byteorder='big'))
    else:  # too many vertices
        f.close()
        return

    val, k = 0, 0
    for i in adjacency_list:
        val = val ^ i
        if k % 6 == 5:
            f.write((val + 63).to_bytes(1, byteorder='big'))
            val = 0
        val = val << 1
        k += 1

    f.close()


def graphml_format(vertex: List[Vertex], save_path: str):
    """
    Saves the graph in .grapml format

    :param save_path: Folder path with the file name. No extension
    :param vertex: Lists of vertex
    :return:

    Args:

    """

    save_path = save_path + '.graphml'
    f = open(save_path, "w")
    namespace = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' \
                + '<graphml' \
                + ' xmlns="http://graphml.graphdrawing.org/xmlns"' \
                + ' xmlns:java="http://www.yworks.com/xml/yfiles-common/1.0/java"' \
                + ' xmlns:sys="http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0"' \
                + ' xmlns:x="http://www.yworks.com/xml/yfiles-common/markup/2.0"' \
                + ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' \
                + ' xmlns:y="http://www.yworks.com/xml/graphml"' \
                + ' xmlns:yed="http://www.yworks.com/xml/yed/3"' \
                + ' xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns ' \
                + 'http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">\n'

    key1 = '<key for="node" id="d0" yfiles.type="nodegraphics"/>\n'
    key2 = '<key for="edge" id="d1" yfiles.type="edgegraphics"/>\n'
    graph_node = '<graph edgedefault="undirected" id="G">\n'

    f.write(namespace)
    f.write(key1)
    f.write(key2)
    f.write(graph_node)

    size = len(vertex)
    adjacency_matrix = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        vertex[i].id = i

    # node definition
    for V in vertex:
        node = '<node id="n' + str(V.id) + '">\n'
        node = node + '<data key="d0">\n'
        node = node + '<y:ShapeNode>\n'
        node = node + '<y:Geometry height="30.0" width="30.0" x="' + str(V.x) + '" y="' + str(V.y) + '"/>\n'  # localization and size
        if V.color == Color.OBJECT:
            node = node + '<y:Fill color="#000000" transparent="false"/>\n'  # filled vertices
        else:
            node = node + '<y:Fill color="#000000" transparent="true"/>\n'  # unfilled vertices
        node = node + '<y:BorderStyle color="#000000" type="line" width="4.0"/>\n'
        node = node + '<y:Shape type="circle"/>\n'
        node = node + '</y:ShapeNode>\n'
        node = node + '</data>\n'
        node = node + '</node>\n'
        f.write(node)

        # completing the neighborhood matrix
        for W in V.neighbour_list:
            adjacency_matrix[V.id][W] = 1

    nr = 0
    for i in range(0, size):
        for j in range(0, i):
            if adjacency_matrix[i][j] == 1:
                edge = '<edge id="e'+str(nr)+'" source="n'+str(i)+'" target="n'+str(j)+'">\n'
                edge = edge + '<data key="d1">\n'
                edge = edge + '<y:PolyLineEdge>\n'
                edge = edge + '<y:LineStyle color="#000000" type="line" width="2.0"/>\n'
                edge = edge + '</y:PolyLineEdge>\n'
                edge = edge + '</data>\n'
                edge = edge + '</edge>\n'
                f.write(edge)
                nr = nr + 1

    f.write("</graph>\n")
    f.write("</graphml>\n")
    f.close()

