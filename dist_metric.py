import torch
import sys
sys.path.append('../')
from train_dist import MatchNet
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
path_param = '../models/match_net_99_30_0.000001_all.pth'
match_net = MatchNet('train')
match_net.load_state_dict(torch.load(path_param))

reorder_for_matchnet= [3, 4, 5, 6, 1,2, 0]
#AB3DMOT: (h, w, l, x, y, z, rot_y)
#matchnet: (x, y, z, rot_y, w, l, h)

dist_th = 5

def learned_dist(bbox1, bbox2):
    if compute_2D_dist(bbox1, bbox2)<dist_th:
        return match_net(torch.Tensor(bbox1).unsqueeze(0), torch.Tensor(bbox2).unsqueeze(0)).item()
    return 100

def compute_2D_dist(bbox1, bbox2):

    return np.linalg.norm( (bbox1-bbox2)[0:3])

def associate_detections_to_trackers_bbox(detections,trackers,iou_threshold=0.1):
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     # ablation study
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)  

  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = learned_dist(det[reorder_for_matchnet],trk[reorder_for_matchnet])   #replaced 3d iou with learned distance 
  matched_indices = linear_assignment(iou_matrix)      # hungarian algorithm

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    #if(iou_matrix[m[0],m[1]]<iou_threshold):
    if(iou_matrix[m[0],m[1]]>0.5):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)