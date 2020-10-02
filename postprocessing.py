from Vertex import Vertex
from typing import List


def postprocess(vertices_list: List[Vertex], save_path: str, is_rotated: bool):
    """
    Save graph in .graph6 abd .graphml formats

    :param vertices_list: list of detected vertices and connections between them
    :param save_path: path for saving resulted files
    :param is_rotated: flag indicating if graph has been rotated in preprocessing, if True rotation must be undone
    """
    graphml_format(vertices_list, save_path, is_rotated)
    graph6_format(vertices_list, save_path)


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
    adjacency_matrix = [[0 for _ in range(size)] for _ in range(size)]
    adjacency_list = []
    for i in range(0, size):
        vertex[i].id = i

    for V in vertex:
        for W in V.adjacency_list:
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


def graphml_format(vertex: List[Vertex], save_path: str, is_rotated: bool):
    """
    Saves the graph in .grapml format

    :param save_path: Folder path with the file name. No extension
    :param vertex: Lists of vertex
    :param is_rotated: flag indicating if graph has been rotated in preprocessing, if True rotation must be undone
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
    adjacency_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(0, size):
        vertex[i].id = i

    # node definition
    for V in vertex:
        node = '<node id="n' + str(V.id) + '">\n'
        node = node + '<data key="d0">\n'
        node = node + '<y:ShapeNode>\n'
        vx, vy = (V.x, V.y) if not is_rotated else (V.y, -V.x)  # if rotation took place inverse rotation
        node = node + '<y:Geometry height="30.0" width="30.0" x="' + str(vx) + '" y="' + str(vy) + '"/>\n'
        node = node + '<y:Fill color="#'+color_string(V.color)+'" transparent="'+('false' if V.is_filled else 'true') + '"/>\n'
        node = node + '<y:BorderStyle color="#000000" type="line" width="4.0"/>\n'
        node = node + '<y:Shape type="ellipse"/>\n'
        node = node + '</y:ShapeNode>\n'
        node = node + '</data>\n'
        node = node + '</node>\n'
        f.write(node)

        # completing the adjacency matrix
        for W in V.adjacency_list:
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


def color_string(color: (int, int, int)):
    """
    Converts BGR integer values into concatenated string of hex values

    :param color: input 3 channels of BGR color
    :return: RGB string of hex values
    """
    hex_string = ""
    for i in range(2, -1, -1):  # reposition channels form BGR to RGB
        if color[i] < 16:
            hex_string += "0"  # add leading zero for one digit hex values
        hex_string += hex(color[i])[2:]  # remove '0x' prefix and concatenate

    return hex_string
