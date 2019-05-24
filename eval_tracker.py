import argparse
import numpy as np
from scipy.spatial.distance import cdist
from statistics import mode
from collections import Counter


def bb_intersection_over_union_by_Adrian_Rosebrock(boxA, boxB):
    """
    Parameters
    ----------
    boxA : 
        An matrix of top-left bottom-right boundingbox
    boxB : 
        An matrix of top-left bottom-right boundingbox
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def process_line(line):

    def tlwh_to_tlbr(bbs):
        '''
        definition:
                (x,y,w,h) --> (x,y,x,y)
        '''
        for bb in bbs:
            bb[2] = bb[2] + bb[0]
            bb[3] = bb[3] + bb[1]
        return bbs

    _all = np.array([int(float(num)) for num in line.split()]).reshape(-1, 5)
    _id = _all[:,:1]
    _bb = _all[:,1:]
    return _id.reshape(1, -1)[0], _bb


def find_most_common(ids):
    try:
        top = mode(ids)
    except:
        top = ids[0]
    return top


def main(gen_file, gt_file):

    """
    Parameters
    ----------
    gen_file :
        A text file with the line content format: <trackid> <top-left x> <top-left y> <bottom-right x> <bottom-right y>
    gt_file :
        A text file with the line content format: <trackid> <top-left x> <top-left y> <bottom-right x> <bottom-right y>
    """
    trackers = {}
    gt_trackers = {}
    with open(gen_file, 'r') as gen:
        with open(gt_file, 'r') as gt:
            for gen_line, gt_line in zip(gen, gt):
                gen_ids, gen_bbs = process_line(gen_line)
                gt_ids, gt_bbs = process_line(gt_line)
                if gt_ids.size > 0 and gen_ids.size > 0:
                    # print(gen_ids)
                    # print(gt_ids)
                    matrix = cdist(gen_bbs, gt_bbs, bb_intersection_over_union_by_Adrian_Rosebrock)
                    # print(matrix)
                    max_matrix = np.argmax(matrix, axis=1)
                    # print(max_matrix)
                    _count = 0
                    for i, gen_id in enumerate(gen_ids):
                        if matrix[i, max_matrix[i]] >= 0.5:
                            _count += 1
                            # if gt_ids[max_matrix[i]] not in [1, 3, 6]:
                            if gt_ids[max_matrix[i]] not in trackers.keys():
                                trackers[gt_ids[max_matrix[i]]] = []
                            trackers[gt_ids[max_matrix[i]]].append(gen_id)
                    # print('Detection accuracy', _count / float(gt_ids.size))
                    # print(gt_trackers)
                    for _id in gt_ids:
                        # if _id in [1, 3, 6]:
                        #     continue
                        if _id not in gt_trackers.keys():
                            gt_trackers[_id] = 0
                        gt_trackers[_id] += 1

    # print(trackers)
    # debug_tracker = trackers[5]
    # print()
    print("ground_truth/frames:", gt_trackers)
    # print(len(debug_tracker))
    # print("num of IDs:", len(list(set(Counter(debug_tracker).elements()))))
    # print("id_max_quan:", Counter(debug_tracker).most_common(7))

    counts = []
    for i in gt_trackers:
        if i in trackers.keys():
            _, _count = Counter(trackers[i]).most_common(1)[0]
        else:
            _count = 0
        acc = _count / float(gt_trackers[i])
        print("Person accuracy", i, acc)
        counts.append(acc)

    print("Tracking accuracy", np.array(counts).mean())


if __name__ == '__main__':
    # Main steps

    parser = argparse.ArgumentParser('For demo only',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f',
                        '--gen',
                        help='file generated from tracker')
    parser.add_argument('-g',
                        '--gt',
                        help='groundtruth file')

    args = parser.parse_args()

    main(args.gen, args.gt)
