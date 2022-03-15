import os
import numpy as np
import imageio 
import cv2

def load_data(basedir= "data/bottles", res=1):
    imgs = [[],[]]
    poses = [[],[]]
    img_root = os.path.join(basedir, "rgb")
    pose_root = os.path.join(basedir, "pose")
    for filename in os.listdir(img_root):
        data_name = filename.split('.')[0]
        if filename.split('.')[1] != 'png':
            continue
        img = imageio.imread(os.path.join(img_root, filename))
        img = (np.array(img) / 255.).astype(np.float32)
        pose = np.loadtxt(os.path.join(pose_root, data_name+".txt"))
        index = int(data_name[0])
        imgs[index].append(img)
        poses[index].append(pose)
    
    H, W = imgs[0][0].shape[:2]
    print(H, W)
    focal = 875.
    K = [[875., 0., 400.],
        [0., 875., 400.],
        [0., 0., 1.]]
        
    H = H//res
    W = W//res
    focal = focal/res

    for i in range(2):
        for j, img in enumerate(imgs[i]):
            imgs[i][j] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    imgs = np.array(imgs)
    poses = np.array(poses)
    print(imgs.shape)
    print(poses.shape)

    return imgs, poses, [H, W, focal], K