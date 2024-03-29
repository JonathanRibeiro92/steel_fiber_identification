import numpy as np
from skimage import measure
from stl import mesh


def generate_mesh(points_array):
    # Crie uma instância vazia de um objeto STL
    vertices = []
    faces = []

    # Adicione os vértices e as faces ao objeto STL
    for i in range(len(points_array)):
        # Adicione cada ponto como um vértice
        vertices.append(points_array[i])

        # Se não for o último ponto, adicione uma face entre este ponto e o próximo
        if i < len(points_array) - 1:
            faces.append([i, i + 1, i])

    # Converta as listas de vértices e faces em arrays numpy
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Crie o objeto mesh
    mesh_object = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_object.vectors[i][j] = vertices[f[j], :]

    return mesh_object

def generate_cube_mesh(data):
    # Create vertices and faces from the 3D image
    vertices, faces, _, _ = measure.marching_cubes(data, 0)
    # Convert to mesh and save
    data_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            data_mesh.vectors[i][j] = vertices[f[j], :]

    return data_mesh


def create_volume_from_points(points):
    # Determinar os limites ao longo dos eixos x, y e z
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    # Determinar as dimensões do volume
    volume_shape = (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)

    # Inicializar o volume com zeros
    volume = np.zeros(volume_shape, dtype=int)

    # Preencher o volume com os pontos
    for point in points:
        x, y, z = point
        volume[x - min_x, y - min_y, z - min_z] = 1

    return volume

