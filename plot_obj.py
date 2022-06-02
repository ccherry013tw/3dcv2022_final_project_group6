import open3d as o3d
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy

glass_align_point_dict = {
    'right_eye'         :   False,
    'left_eye'          :   False,
    'center'            :   True,
    'right_eyebrown'    :   True,
    'left_eyebrown'     :   True,
    'right_eye_bottom'  :   False,
    'left_eye_bottom'   :   False,
    'rightmost'         :   False,
    'leftmost'          :   False,
}

open3d_display_dict = {
    'object_pcd'            :   True,
    'face_pcd'              :   True,
    'object_landmark'       :   True, 
    'face_align_landmark'   :   True, 
    'face_all_landmarks'    :   False,
}

def estimateRT(object_points, face_points):

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

def visualize_cv(points1, points2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(points1.shape[0]):
        e = points1[i]
        m = points2[i]
        ax.scatter(e[0], e[1], e[2], c='#006633', marker='^') #Green
        ax.text(e[0], e[1], e[2], i)
        ax.scatter(m[0], m[1], m[2], c='#202020', marker='o') #Black
        ax.text(m[0], m[1], m[2], i)

    plt.show()

def visualize(object_pcd, face_pcd, object_landmark, face_align_landmark, face_all_landmarks):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if open3d_display_dict['object_pcd']:
        vis.add_geometry(object_pcd)

    if open3d_display_dict['face_pcd']:
        vis.add_geometry(face_pcd)

    if open3d_display_dict['object_landmark']:
        vis.add_geometry(object_landmark)

    if open3d_display_dict['face_align_landmark']:
        vis.add_geometry(face_align_landmark)

    if open3d_display_dict['face_all_landmarks']:
        vis.add_geometry(face_all_landmarks)
    
    vis.run()
    vis.destroy_window()

def loadImg(img_id):
    pcd = o3d.geometry.PointCloud()
    img = cv2.imread(f'images/3dcv/face_{img_id}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_dep = np.load(f'images/3dcv_dmp/face_{img_id}.npy')
    colors = []
    points = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_dep[i,j] < 1000 and img_dep[i,j] > 0:
                points.append([i,j,img_dep[i,j]])   
                colors.append(img[i,j]/255)
    points = np.array(points, dtype='f')
    max_depth = np.max(points[:,2])
    points[:,2] = (max_depth - points[:,2]) 

    colors = np.array(colors, dtype='f')

    face_landmarks = np.around(np.load(f'images/3dcv_pred_coord/face_{img_id}.npy')).astype(int)
    face_landmarks[:, [1, 0]] = face_landmarks[:, [0, 1]]

    face_landmarks_3d = []
    for i,j in face_landmarks:
        face_landmarks_3d.append([i,j, max_depth - img_dep[i,j]])

    face_landmarks_3d = np.array(face_landmarks_3d)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, face_landmarks_3d

def load_object_landmark(vertices):
    mark = o3d.geometry.PointCloud()

    idx = np.loadtxt(f'models/{object_name}_pt.txt').astype(int)
    points = [vertices[i] for i in idx]

    mark.paint_uniform_color([1, 0, 0])

    mark.points = o3d.utility.Vector3dVector(points)

    return mark, np.array(points)

def filter_align_points(all_marks):
    use_marks = []
    idx = 0
    for key in glass_align_point_dict:
        if glass_align_point_dict[key]:
            use_marks.append(all_marks[idx])
        idx += 1

    use_marks = np.array(use_marks)

    return use_marks

def get_object_align_points(marks):
    right_eye = (marks[1] + marks[2])/2
    right_eye[2] = 0
    left_eye = (marks[4] + marks[5])/2
    left_eye[2] = 0
    marks[7,2] = -5
    marks[8,2] = 0
    marks[11,2] = 0
    marks[12,2] = 0
    marks[1,2] = 0
    marks[5,2] = 0

    all_marks = np.array([right_eye, left_eye, [0,0,0], marks[7], marks[8], marks[11], marks[12], marks[1], marks[5]])
    
    return filter_align_points(all_marks)


def get_face_align_points(face_landmarks):

    idx_arr = [96,97,51,40,48]
    v = face_landmarks[97] - face_landmarks[96]
    nose = face_landmarks[53]

    right_z = nose[2] - v[2]/2
    right_eye = [face_landmarks[96,0], face_landmarks[96,1], right_z]
    right_eyebrown = [face_landmarks[41,0], face_landmarks[41,1], right_z]
    right_eye_bottom = [face_landmarks[54,0], face_landmarks[66,1], right_z]
    rightmost = right_eye - v/2 

    left_z = nose[2] + v[2]/2
    left_eye = [face_landmarks[97,0], face_landmarks[97,1], left_z]
    left_eyebrown = [face_landmarks[47,0], face_landmarks[47,1], left_z]
    left_eye_bottom = [face_landmarks[54,0], face_landmarks[74,1], left_z]
    leftmost = left_eye + v/2 

    nose_51 = [face_landmarks[51,0], face_landmarks[51,1], nose[2]]

    all_marks =  np.array([right_eye, left_eye, nose_51, right_eyebrown, left_eyebrown, right_eye_bottom, left_eye_bottom, rightmost, leftmost])
    
    return filter_align_points(all_marks)
    
def move_object(R, T, mesh):
    new_vertices =[]
    for pt in np.array(mesh.points):
        new_vertices.append(R@pt+T)
    mesh.points = o3d.utility.Vector3dVector(new_vertices)
    return mesh

def write_Img(img_id, face_pcd, object_pcd):
    points = np.around(np.append(np.asarray(face_pcd.points), np.asarray(object_pcd.points), axis=0)).astype(int)
    colors = np.around(np.append(np.asarray(face_pcd.colors), np.asarray(object_pcd.colors), axis=0)*255).astype(int)
    img = cv2.imread(f'images/3dcv/face_{img_id}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points_color = np.append(points, colors, axis=1)
    points_color = points_color[points_color[:, 0].argsort()]
    points_color = points_color[points_color[:, 1].argsort()]
    points_color = points_color[points_color[:, 2].argsort()]

    for ptc in points_color:
        img[ptc[0], ptc[1],:] = ptc[3:]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'images/img_out/{object_name}/face_{img_id}.jpg', img)

def calculate_glass_loss(est_pts, align_pts):
    diff = est_pts - align_pts
    loss = np.mean(np.linalg.norm(diff, axis = 1))
    
    return loss < 8

def is_side_view(face_landmarks):
    right_ear = face_landmarks[3]
    nose = face_landmarks[54]
    left_ear = face_landmarks[29]

    right_dist = nose[1] - right_ear[1]
    left_dist = left_ear[1] - nose[1]


    if right_dist > left_dist:
        ratio = left_dist/right_dist
        return ratio, 'r'
    else:
        ratio = right_dist/left_dist
        return ratio, 'l'

def get_ear_align_points(face_landmarks, side, ratio, object_num):
    if side == 'r':
        center = copy.deepcopy(face_landmarks[4])
        center[1] -= 10
        v = np.array([0,-1, 0])*ratio + np.array([0,0,-1])*(1-ratio)
    else:
        center = copy.deepcopy(face_landmarks[28])
        center[1] += 10
        v = np.array([0, 1, 0])*ratio + np.array([0,0,-1])*(1-ratio)

    back = copy.deepcopy(center) + 10*v
    bottom = copy.deepcopy(center)
    if object_num == 1:
        bottom[0] += 10
    else:
        bottom[0] += 30

    return np.array([ back, center, bottom])

def make_pcd(points):

    pcd = o3d.geometry.PointCloud()
    pcd.paint_uniform_color([0, 0, 1])
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def plot_glasses(img_id, object_org):
    object_pcd = copy.deepcopy(object_org)
    vertices = np.asarray(object_pcd.points).copy()

    face_pcd, face_all_landmarks = loadImg(img_id)
    
    face_align_points = get_face_align_points(face_all_landmarks)

    face_landmarks_pcd = make_pcd(face_align_points)

    object_landmarks_pcd, marks = load_object_landmark(vertices)

    object_align_points = get_object_align_points(marks)

    est_pts, R, T = estimateRT(object_align_points, face_align_points)

    if calculate_glass_loss(est_pts, face_align_points):

        object_pcd = move_object(R,T, object_pcd)

        object_landmarks_pcd = move_object(R,T, object_landmarks_pcd)

        face_all_landmarks_pcd = make_pcd(face_all_landmarks)

        visualize(object_pcd, face_pcd, object_landmarks_pcd, face_landmarks_pcd, face_all_landmarks_pcd)

        write_Img(img_id, face_pcd, object_pcd)
    else:
        img = cv2.imread(f'images/3dcv/face_{img_id}.jpg')
        cv2.imwrite(f'images/img_out/{object_name}/face_{img_id}.jpg', img)


def plot_earring(img_id, object_org, object_num):
    object_pcd = copy.deepcopy(object_org)
    vertices = np.asarray(object_pcd.points).copy()

    face_pcd, face_all_landmarks = loadImg(img_id)

    ratio, side = is_side_view(face_all_landmarks)

    if ratio < 0.5:
        ear_align_points = get_ear_align_points(face_all_landmarks, side, ratio, object_num)

        object_landmarks_pcd, marks = load_object_landmark(vertices)

        est_pts, R, T = estimateRT(marks, ear_align_points)

        object_pcd = move_object(R,T, object_pcd)

        object_landmarks_pcd = move_object(R,T, object_landmarks_pcd)

        ear_landmarks_pcd = make_pcd(ear_align_points)

        face_all_landmarks_pcd = make_pcd(face_all_landmarks)

        visualize(object_pcd, face_pcd, object_landmarks_pcd, ear_landmarks_pcd, face_all_landmarks_pcd)

        write_Img(img_id, face_pcd, object_pcd)
    else:
        img = cv2.imread(f'images/3dcv/face_{img_id}.jpg')
        cv2.imwrite(f'images/img_out/{object_name}/face_{img_id}.jpg', img)


def plot(img_id, object_name):
    object_org = o3d.io.read_point_cloud(f'models/{object_name}.pcd')
    object_type, object_num = object_name.split('_')
    object_num = int(object_num)
    if img_id < 0:
        for img_id in range(400):
            print(f"Img_{img_id}")
            if object_type == 'glasses':
                plot_glasses(img_id, object_org)
            elif object_type == 'earring':
                plot_earring(img_id, object_org, object_num)
    else:
        print(f"Img_{img_id}")
        if object_type == 'glasses':
            plot_glasses(img_id, object_org)
        elif object_type == 'earring':
            plot_earring(img_id, object_org, object_num)


if __name__ == '__main__':
    object_name = sys.argv[1]
    img_id = int(sys.argv[2])
    plot(img_id, object_name)