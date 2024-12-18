# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import pdb
import torch
import sys

sys.path.insert(0, os.path.expanduser("/home/cby/cogvideo_code/"))
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.config import CfgNode as CN

sys.path.insert(0, 'third_party/CenterNet2/')
# from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

def add_centernet_config(cfg):
    _C = cfg

    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.CENTERNET.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.CENTERNET.PRIOR_PROB = 0.01
    _C.MODEL.CENTERNET.INFERENCE_TH = 0.05
    _C.MODEL.CENTERNET.CENTER_NMS = False
    _C.MODEL.CENTERNET.NMS_TH_TRAIN = 0.6
    _C.MODEL.CENTERNET.NMS_TH_TEST = 0.6
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 100
    _C.MODEL.CENTERNET.NORM = "GN"
    _C.MODEL.CENTERNET.USE_DEFORMABLE = False
    _C.MODEL.CENTERNET.NUM_CLS_CONVS = 4
    _C.MODEL.CENTERNET.NUM_BOX_CONVS = 4
    _C.MODEL.CENTERNET.NUM_SHARE_CONVS = 0
    _C.MODEL.CENTERNET.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.CENTERNET.SIGMOID_CLAMP = 1e-4
    _C.MODEL.CENTERNET.HM_MIN_OVERLAP = 0.8
    _C.MODEL.CENTERNET.MIN_RADIUS = 4
    _C.MODEL.CENTERNET.SOI = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
    _C.MODEL.CENTERNET.POS_WEIGHT = 1.
    _C.MODEL.CENTERNET.NEG_WEIGHT = 1.
    _C.MODEL.CENTERNET.REG_WEIGHT = 2.
    _C.MODEL.CENTERNET.HM_FOCAL_BETA = 4
    _C.MODEL.CENTERNET.HM_FOCAL_ALPHA = 0.25
    _C.MODEL.CENTERNET.LOSS_GAMMA = 2.0
    _C.MODEL.CENTERNET.WITH_AGN_HM = False
    _C.MODEL.CENTERNET.ONLY_PROPOSAL = False
    _C.MODEL.CENTERNET.AS_PROPOSAL = False
    _C.MODEL.CENTERNET.IGNORE_HIGH_FP = -1.
    _C.MODEL.CENTERNET.MORE_POS = False
    _C.MODEL.CENTERNET.MORE_POS_THRESH = 0.2
    _C.MODEL.CENTERNET.MORE_POS_TOPK = 9
    _C.MODEL.CENTERNET.NOT_NORM_REG = True
    _C.MODEL.CENTERNET.NOT_NMS = False
    _C.MODEL.CENTERNET.NO_REDUCE = False

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/lvis/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.OUT_FEATURES = ['dla2']
    _C.MODEL.DLA.USE_DLA_UP = True
    _C.MODEL.DLA.NUM_LAYERS = 34
    _C.MODEL.DLA.MS_OUTPUT = False
    _C.MODEL.DLA.NORM = 'BN'
    _C.MODEL.DLA.DLAUP_IN_FEATURES = ['dla3', 'dla4', 'dla5']
    _C.MODEL.DLA.DLAUP_NODE = 'conv'

    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
    _C.INPUT.NOT_CLAMP_BOX = False
    
    _C.DEBUG = False
    _C.SAVE_DEBUG = False
    _C.SAVE_PTH = False
    _C.VIS_THRESH = 0.3
    _C.DEBUG_SHOW_NAME = False

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        if args.webcam == "screen":
            cam = ScreenGrab()
        else:
            cam = cv2.VideoCapture(int(args.webcam))
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        
        #my code
        class_name = torch.load("/home/cby/cogvideo_code/Detic/demo_class_name.pt")
        #target_class = "person"
        #target_calss = "bird"
        #target_class = "car_(automobile)
        target_class = "cat"
        #mask_save_path = "/home/whl/workspace/cogvideo_edit/metrics_mask/chid.pt"
        #mask_save_path = "/home/whl/workspace/cogvideo_edit/metrics_mask/swan.pt"
        #mask_save_path = "/home/whl/workspace/cogvideo_edit/metrics_mask/car_turn.pt"
        mask_save_path = "/home/cby/cogvideo_code/mask/cat_flower.pt"
        all_mask = []
    
        for predictions, vis_frame in tqdm.tqdm(demo.run_on_video_msl(video), total=num_frames):
            
            # #target category
            # mask = torch.zeros([480,720],dtype = torch.bool)
            # for j in range(len(predictions.pred_classes)):
            #     if class_name[predictions.pred_classes[j]] == "carrot" or \
            #        class_name[predictions.pred_classes[j]] == "cat" or \
            #        class_name[predictions.pred_classes[j]] == "rabbit" or \
            #        class_name[predictions.pred_classes[j]] == "ferret" or \
            #        class_name[predictions.pred_classes[j]] == "hamster" :
            #        mask = mask | predictions.pred_masks[j] 
            # all_mask.append(mask)
        
            #target category
            mask = torch.zeros([480,720],dtype = torch.bool)
            for j in range(len(predictions.pred_classes)):
                if class_name[predictions.pred_classes[j]] == target_class: 
                   mask = predictions.pred_masks[j] 
            all_mask.append(mask)
            
            # #all category
            # mask = torch.zeros([480,720],dtype = torch.bool)
            # for j in range(len(predictions.pred_classes)):
            #     mask = mask | predictions.pred_masks[j]
            # all_mask.append(mask)


            if args.output:
                output_file.write(vis_frame)
                
            # else:
            #     cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            #     cv2.imshow(basename, vis_frame)
            #     if cv2.waitKey(1) == 27:
            #         break  # esc to quit

        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
        
        all_mask = torch.stack(all_mask)
        torch.save(all_mask, mask_save_path)
    pdb.set_trace()
