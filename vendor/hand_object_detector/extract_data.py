# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
"""this model extracts data from a video and saves it to a json file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
import glob
import multiprocessing as mp
import queue
import threading
from collections import defaultdict
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import json
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--input_folder', dest='input_folder',
                      help='path to folder containing video files',
                      required=True, type=str)
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true', default=True)
  parser.add_argument('--num_gpus', dest='num_gpus',
                      help='number of GPUs to use (0 for auto-detect, -1 for CPU only)',
                      default=0, type=int)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=False)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def get_video_files(folder_path):
  """
  Get all video files from the specified folder
  """
  video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.m4v']
  video_files = []
  
  for extension in video_extensions:
    pattern = os.path.join(folder_path, extension)
    video_files.extend(glob.glob(pattern))
    # Also check for uppercase extensions
    pattern = os.path.join(folder_path, extension.upper())
    video_files.extend(glob.glob(pattern))
  
  return sorted(video_files)

def get_available_gpus():
  """
  Get the number of available GPUs
  """
  if not torch.cuda.is_available():
    return 0
  return torch.cuda.device_count()

def initialize_model_on_gpu(args, gpu_id):
  """
  Initialize the model on a specific GPU
  """
  # Set the device
  torch.cuda.set_device(gpu_id)
  
  # Load configuration
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda
  np.random.seed(cfg.RNG_SEED)

  # load model
  model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
  if not os.path.exists(model_dir):
    raise Exception('There is no input directory for loading network from ' + model_dir)
  load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

  # initialize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    raise ValueError("network is not defined")

  fasterRCNN.create_architecture()

  print(f"[GPU {gpu_id}] Loading checkpoint {load_name}")
  checkpoint = torch.load(load_name, map_location=f'cuda:{gpu_id}')
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print(f'[GPU {gpu_id}] Model loaded successfully!')

  # Move model to GPU
  fasterRCNN.cuda(gpu_id)
  fasterRCNN.eval()

  # Initialize tensor holders
  im_data = torch.FloatTensor(1).cuda(gpu_id)
  im_info = torch.FloatTensor(1).cuda(gpu_id)
  num_boxes = torch.LongTensor(1).cuda(gpu_id)
  gt_boxes = torch.FloatTensor(1).cuda(gpu_id)
  box_info = torch.FloatTensor(1).cuda(gpu_id)

  return fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info, pascal_classes

def gpu_worker(gpu_id, video_queue, result_queue, args):
  """
  Worker function that processes videos on a specific GPU
  """
  try:
    # Initialize model on this GPU
    fasterRCNN, im_data, im_info, num_boxes, gt_boxes, box_info, pascal_classes = initialize_model_on_gpu(args, gpu_id)
    
    cfg.CUDA = True
    
    with torch.no_grad():
      while True:
        try:
          # Get next video from queue (with timeout to avoid hanging)
          video_path = video_queue.get(timeout=1)
          if video_path is None:  # Sentinel value to stop worker
            break
            
          # print(f"[GPU {gpu_id}] Processing: {os.path.basename(video_path)}")
          
          # Process the video
          start_time = time.time()
          success = process_single_video_gpu(video_path, fasterRCNN, args, im_data, im_info, 
                                           num_boxes, gt_boxes, box_info, pascal_classes, gpu_id)
          end_time = time.time()
          
          # Report result
          result_queue.put({
            'gpu_id': gpu_id,
            'video_path': video_path,
            'success': success,
            'processing_time': end_time - start_time
          })
          
          # print(f"[GPU {gpu_id}] Completed: {os.path.basename(video_path)} in {end_time - start_time:.2f}s")
          
        except queue.Empty:
          continue
        except Exception as e:
          print(f"[GPU {gpu_id}] Error processing video: {str(e)}")
          result_queue.put({
            'gpu_id': gpu_id,
            'video_path': video_path if 'video_path' in locals() else 'unknown',
            'success': False,
            'processing_time': 0,
            'error': str(e)
          })
          
  except Exception as e:
    print(f"[GPU {gpu_id}] Worker initialization failed: {str(e)}")
    result_queue.put({
      'gpu_id': gpu_id,
      'video_path': 'initialization',
      'success': False,
      'processing_time': 0,
      'error': str(e)
    })

def process_single_video_gpu(video_path, fasterRCNN, args, im_data, im_info, num_boxes, gt_boxes, box_info, pascal_classes, gpu_id):
  """
  Process a single video file on a specific GPU
  """
  # Video file mode
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print(f"[GPU {gpu_id}] Warning: Could not open video file: {video_path}")
      return False
  
  # Get video properties
  fps = 30
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  frame_count = 0
  video_data = []

  thresh_hand = args.thresh_hand 
  thresh_obj = args.thresh_obj
  vis = args.vis

  try:
    while True:
        frame_count += 1

        # Get frame from video
        ret, frame = cap.read()
        if not ret:
            break
        im_in = np.array(frame)
        
        # bgr
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extract predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(gpu_id) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda(gpu_id)
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(gpu_id) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda(gpu_id)
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        
        if vis:
            im2show = np.copy(im)
        obj_dets, hand_dets = None, None
        for j in xrange(1, len(pascal_classes)):
            if pascal_classes[j] == 'hand':
              inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
              inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              if pascal_classes[j] == 'targetobject':
                obj_dets = cls_dets.cpu().numpy()
              if pascal_classes[j] == 'hand':
                hand_dets = cls_dets.cpu().numpy()
              
        if vis:
          # visualization
          im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
          
        # extract hand info
        frame_data = {"left_hand": None, "right_hand": None, "objects": None}
        if hand_dets is not None:
          for i in range(len(hand_dets)):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = float(hand_dets[i, 4])  # Convert numpy float32 to Python float
            lr = hand_dets[i, -1] # 0 means left, 1 means right
            state = hand_dets[i, 5]
            state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}
            state = state_map2[state]
            if lr == 0 and score > args.thresh_hand:
              frame_data["left_hand"] = {"bbox": bbox, "score": score, "state": state}
            elif lr == 1 and score > args.thresh_hand:
              frame_data["right_hand"] = {"bbox": bbox, "score": score, "state": state}
        # extract object info
        if obj_dets is not None:
          for i in range(len(obj_dets)):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = float(obj_dets[i, 4])  # Convert numpy float32 to Python float
            if score > args.thresh_obj:
              frame_data["objects"] = {"bbox": bbox, "score": score}
        video_data.append(frame_data)

        # Progress output (less frequent to avoid spam)
        # if frame_count % 30 == 0 or frame_count >= total_frames:
        #     print(f'[GPU {gpu_id}] {os.path.basename(video_path)}: frame {frame_count}/{total_frames}')
        
        # Check if we've processed all frames
        if frame_count >= total_frames:
            break
              
  except Exception as e:
    print(f"[GPU {gpu_id}] Error during video processing: {str(e)}")
    cap.release()
    return False
            
  # Cleanup
  cap.release()
  
  # save video_data to json
  output_json_path = f'{video_path.rsplit(".", 1)[0]}.json'
  with open(output_json_path, 'w') as f:
    json.dump(video_data, f)
  # print(f'[GPU {gpu_id}] Video processing complete. Output saved to: {output_json_path}')
  return True

if __name__ == '__main__':
  # Set multiprocessing start method to 'spawn' for CUDA compatibility
  mp.set_start_method('spawn', force=True)
  
  args = parse_args()

  # Get all video files from the input folder
  video_files = get_video_files(args.input_folder)
  if not video_files:
      print(f"No video files found in folder: {args.input_folder}")
      print("Supported formats: mp4, avi, mov, mkv, flv, wmv, m4v")
      sys.exit(1)
  
  print(f"Found {len(video_files)} video files to process:")
  # for video_file in video_files:
  #     print(f"  - {os.path.basename(video_file)}")

  # Determine number of GPUs to use
  available_gpus = get_available_gpus()
  
  if args.num_gpus == -1:  # CPU only
    num_gpus = 0
    args.cuda = False
  elif args.num_gpus == 0:  # Auto-detect
    num_gpus = available_gpus
  else:  # User specified
    num_gpus = min(args.num_gpus, available_gpus)
  
  print(f"\nGPU Configuration:")
  print(f"  Available GPUs: {available_gpus}")
  print(f"  Using GPUs: {num_gpus}")
  
  if num_gpus == 0:
    print("Warning: Running on CPU only. This will be much slower.")
    # Fall back to single-threaded CPU processing
    # ... (implement CPU fallback if needed)
    sys.exit(1)
  
  # Create queues for communication between processes
  video_queue = mp.Queue()
  result_queue = mp.Queue()
  
  # Add all videos to the queue
  for video_path in video_files:
    video_queue.put(video_path)
  
  # Add sentinel values to signal workers to stop
  for _ in range(num_gpus):
    video_queue.put(None)
  
  # Start worker processes
  processes = []
  for gpu_id in range(num_gpus):
    p = mp.Process(target=gpu_worker, args=(gpu_id, video_queue, result_queue, args))
    p.start()
    processes.append(p)
  
  # Monitor progress
  start_time = time.time()
  completed_videos = 0
  successful_videos = 0
  failed_videos = 0
  gpu_stats = defaultdict(lambda: {'count': 0, 'time': 0})
  
  print(f"\nStarting parallel processing with {num_gpus} GPUs...")
  
  # Create progress bar
  progress_bar = tqdm(
    total=len(video_files),
    desc="Processing videos",
    unit="video",
    ncols=100,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix[0]} Failed: {postfix[1]}',
    postfix=[0, 0]
  )
  
  # Collect results
  while completed_videos < len(video_files):
    try:
      result = result_queue.get(timeout=5)
      completed_videos += 1
      
      gpu_id = result['gpu_id']
      video_name = os.path.basename(result['video_path'])
      
      if result['success']:
        successful_videos += 1
        gpu_stats[gpu_id]['count'] += 1
        gpu_stats[gpu_id]['time'] += result['processing_time']
        progress_bar.set_description(f"✓ {video_name[:20]:<20}")
      else:
        failed_videos += 1
        error_msg = result.get('error', 'Unknown error')
        progress_bar.set_description(f"✗ {video_name[:20]:<20}")
        
      # Update progress bar
      progress_bar.postfix[0] = successful_videos
      progress_bar.postfix[1] = failed_videos
      progress_bar.update(1)
        
    except queue.Empty:
      progress_bar.set_description("Waiting for results...")
      continue
  
  # Close progress bar
  progress_bar.close()
  
  # Wait for all processes to complete
  for p in processes:
    p.join()
  
  # Print final statistics
  total_time = time.time() - start_time
  print("\n" + "="*80)
  print("PROCESSING COMPLETE")
  print("="*80)
  print(f"Total videos: {len(video_files)}")
  print(f"Successful: {successful_videos}")
  print(f"Failed: {failed_videos}")
  print(f"Total wall time: {total_time:.2f} seconds")
  print(f"Average wall time per video: {total_time/len(video_files):.2f} seconds")
  
  print(f"\nGPU Statistics:")
  for gpu_id in range(num_gpus):
    stats = gpu_stats[gpu_id]
    if stats['count'] > 0:
      avg_time = stats['time'] / stats['count']
      print(f"  GPU {gpu_id}: {stats['count']} videos, avg {avg_time:.1f}s per video")
    else:
      print(f"  GPU {gpu_id}: 0 videos processed")
  
  if successful_videos > 0:
    total_processing_time = sum(gpu_stats[gpu_id]['time'] for gpu_id in range(num_gpus))
    speedup = total_processing_time / total_time
    print(f"\nSpeedup achieved: {speedup:.1f}x (vs sequential processing)")