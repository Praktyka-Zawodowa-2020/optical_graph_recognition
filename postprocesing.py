from Vertex import Vertex


def graph6_format(vertex: List[Vertex]):
    """
    Saves the graph in .graph6 format
    :param vertex: Lists of vertex
    :return:
    """
    size = len(vertex)
    adjacency_matrix = [[0 for x in range(size)] for y in range(size)]
    adjacency_list = []
    for i in range(0, size):
        vertex[i].id = i

    for V in vertex:
        for W in V.neighbour_list:
            adjacency_matrix[V.id][W.id] = 1

    for i in range(0, size):
        for j in range(0, i):
            adjacency_list.append(adjacency_matrix[j][i])

    #align on the right with 0 so that the length is a multiple of 6.
    if len(adjacency_list) % 6 != 0:
        while len(adjacency_list) % 6 != 0:
            adjacency_list.append(0)

    f = open("graph.g6", "wb")
    if size < 63:
        f.write((size + 63).to_bytes(1, byteorder='big'))
    elif 63 <= size <= 258047:
        f.write((126).to_bytes(1, byteorder='big'))
    elif 258047 <= size <= 68719476735:
        f.write((126).to_bytes(1, byteorder='big'))
        f.write((126).to_bytes(1, byteorder='big'))
    else:
        f.close()
        print("too many vertices")
        return

    val, k = 0, 0
    for i in adjacency_list:
        val = val ^ i
        val = val << 1
        k += 1
        if k % 6 == 5:
            f.write((val + 63).to_bytes(1, byteorder='big'))
            val = 0

    f.close()
