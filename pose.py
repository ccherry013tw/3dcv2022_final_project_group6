import matplotlib.pyplot as plt
import numpy as np

def main(object_points, face_points):

    face_points = face_points.astype('float64')
    object_points = object_points.astype('float64')

    face_centroid = np.average(face_points, axis=0)
    object_centroid = np.average(object_points, axis=0)

    
    face_points -= face_centroid
    object_points -= object_centroid

    h = face_points.T @ object_points
    u, _, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    r = v @ e @ u.T

    # calculate T
    object_points += object_centroid
    trans_points = object_points @ r
    trans_centroid = np.average(trans_points, axis=0)

    t = face_centroid - trans_centroid
    estimated_points = trans_points @ r + t
    
    return estimated_points, r, t

def visualize(points1, points2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(points1.shape[0]):
        e = points1[i]
        m = points2[i]
        ax.scatter(e[0], e[1], e[2], c='#006633', marker='^') #Green
        ax.text(e[0], e[1], e[2], i)
        ax.scatter(m[0], m[1], m[2], c='#202020', marker='o') #Black
        ax.text(m[0], m[1], m[2], i)

    plt.savefig(f'plot.png')

if __name__ == "__main__":
    points1 = np.array([[-1, 1, 0], [0, 0, 0]])
    points2 = np.array([[-3, -1, -2], [-2, -2, -2]])
    
    estimated_points, _, _ = main(points1, points2)
    visualize(estimated_points, points2)