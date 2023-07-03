import os
import json
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN


def convert_bbox_hair(mode):
    path = 'data/' + mode + '/json/'
    filelist = glob(path + '*.json')
    for file in filelist:
        with open(file, 'r') as f:
            data = json.load(f)
            coordinates = data['shapes']
            image_width = data['imageWidth']
            image_height = data['imageHeight']
            
        name = str(os.path.basename(file).replace('.json', '.txt'))
        
        with open("yolo_hair/" + mode + "/" + name, 'w') as f:
            for i, coord in enumerate(coordinates):
                points = [xy for xy in coord['points']]
                points = np.transpose(points)
                x_min, x_max = np.amin(points[0]), np.amax(points[0])
                y_min, y_max = np.amin(points[1]), np.amax(points[1])
                xc = float((x_min + x_max) / 2 / image_width)
                yc = float((y_min + y_max) / 2 / image_height)
                w = float((x_max - x_min) / image_width)
                h = float((y_max - y_min) / image_height)
                f.write(f"0 {xc} {yc} {w} {h}\n")


def clustering(data, eps):
    dbscan = DBSCAN(min_samples=1, eps=eps)
    dbscan.fit(data)
    labels = dbscan.labels_

    cluster = list()
    w_and_h = list()
    for idx in np.unique(labels):
        clustered = [pts for pts, lbs in zip(data, labels) if lbs == idx]
        cluster.append(np.mean(clustered, axis=0))
        w_and_h.append(np.std(clustered, axis=0))

    return cluster, w_and_h


def convert_bbox_root(mode, eps=75):
    path = 'data/' + mode + '/json/'
    filelist = glob(path + '*.json')
    for file in filelist:
        with open(file, 'r') as f:
            data = json.load(f)
            image_width = data['imageWidth']
            image_heights = data['imageHeight']

        name = str(os.path.basename(file).replace('.json', '.txt'))

        roots = list()
        with open("yolo_root/" + mode + "/" + name, 'w') as f:
            for hair in data['shapes']:
                roots.append(hair['points'][0])

            cluster, w_and_h = clustering(data=roots, eps=eps)
            for (x, y), (w, h) in zip(cluster, w_and_h):
                w += eps
                h += eps
                f.write(f"0 {(x - w/2)/image_width} {(y - h/2)/image_heights} {w/image_width} {h/image_heights}\n")


if __name__ == '__main__':
    modes = ['train', 'val', 'test']
    for mode in modes:
        # convert_bbox_hair(mode)
        convert_bbox_root(mode)
