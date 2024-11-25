import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
from datetime import datetime
from PIL import Image
from mindyolo.utils.config import load_config, Config
from mindyolo.models import create_model
from mindyolo.utils.utils import  set_seed
import mindspore as ms
from mindspore import Tensor, context, nn
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils import logger
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh, process_mask_upsample, scale_image

from prompt import FastSAMPrompt

def segment(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 60.0,
    img_size: int = 1024,
    stride: int = 16,
    num_class: int = 1,
    is_coco_dataset: bool = True,
):
    # Resize
    h_ori, w_ori = img.shape[:2]  # orig hw
    r = img_size / max(h_ori, w_ori)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

    # Transpose Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    # Run infer
    _t = time.time()
    out, (_, _, prototypes) = network(imgs_tensor)  # inference and training outputs
    infer_times = time.time() - _t

    # Run NMS
    t = time.time()
    _c = num_class + 4 if conf_free else num_class + 5
    out = out.asnumpy()
    bboxes, mask_coefficient = out[:, :, :_c], out[:, :, _c:]
    out = non_max_suppression(
        bboxes,
        mask_coefficient,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t

    prototypes = prototypes.asnumpy()

    result_dict = {"category_id": [], "bbox": [], "score": [], "segmentation": []}
    total_category_ids, total_bboxes, total_scores, total_seg = [], [], [], []
    for si, (pred, proto) in enumerate(zip(out, prototypes)):
        if len(pred) == 0:
            continue

        # Predictions
        pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=imgs_tensor[si].shape[1:])
        pred_masks = pred_masks.astype(np.float32)
        pred_masks = scale_image((pred_masks.transpose(1, 2, 0)), (h_ori, w_ori))
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # native-space pred

        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        category_ids, bboxes, scores, segs = [], [], [], []
        for ii, (p, b) in enumerate(zip(pred.tolist(), box.tolist())):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))
            segs.append(pred_masks[:, :, ii])

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)
        total_seg.extend(segs)

    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)
    result_dict["segmentation"].extend(total_seg)

    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is:")
    for k, v in result_dict.items():
        if k == "segmentation":
            logger.info(f"{k} shape: {v[0].shape}")
        else:
            logger.info(f"{k}: {v}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict


def infer(args, image, network):
    # Init
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
    image = Image.open(args.image_path)
    image = np.array(image)

    # Detect
    is_coco_dataset = "coco" in args.data.dataset_name
    result_dict = segment(
        network=network,
        img=image,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        nms_time_limit=args.nms_time_limit,
        img_size=args.img_size,
        stride=max(max(args.network.stride), 32),
        num_class=args.data.nc,
        is_coco_dataset=is_coco_dataset,
    )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(image, result_dict,)

    if args.point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args.point_prompt, pointlabel=args.point_label
        )
        points = args.point_prompt
        point_label = args.point_label

    elif args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
    
    else:
        ann = prompt_process.everything_prompt()

    im = prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.image_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )
    # if args.save_result:
    #     save_path = os.path.join(args.save_dir, "segment_results")
    #     im = draw_result(image, result_dict, args.data.names, is_coco_dataset=is_coco_dataset, save_path=save_path)
    logger.info("Infer completed.")
    return im


def draw_result(image, result_dict, data_names, is_coco_dataset=True, save_path="./detect_results"):
    import random
    import cv2
    from mindyolo.data import COCO80_TO_COCO91_CLASS

    os.makedirs(save_path, exist_ok=True)
    save_result_path = os.path.join(save_path, "result.jpg")
    im = image
    category_id, bbox, score = result_dict["category_id"], result_dict["bbox"], result_dict["score"]
    seg = result_dict.get("segmentation", None)
    mask = None if seg is None else np.zeros_like(im, dtype=np.float32)
    for i in range(len(bbox)):
        # draw box
        x_l, y_t, w, h = bbox[i][:]
        x_r, y_b = x_l + w, y_t + h
        x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
        if seg:
            _color_seg = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
            mask += seg[i][:, :, None] * _color_seg[None, None, :]

    # save results
    if seg:
        im = (0.5 * im + 0.5 * mask).astype(np.uint8)
    return im

def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--weight", type=str, default="./weight/FastSAM-x.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--image_path", type=str, default="", help="image path(s)")
    parser.add_argument("--img_size", type=int, default=1024, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.9, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")

    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")

    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument("--config", type=str, default="", help="YAML config file specifying default arguments.")
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument("--point_prompt", type=str, default="[[0,0]]", help="[x1,y1],[x2,y2]")
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    return parser

def parse_args_for_infer(parser):
    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="./config/fastsam-x.yaml", help="YAML config file specifying default arguments."
    )
    args_config, remaining = parser_config.parse_known_args()
    # Do we have a config file to parse?
    if args_config.config:
        cfg, _, _ = load_config(args_config.config)
        cfg = Config(cfg)
        parser.set_defaults(**cfg)
        parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return Config(vars(args))

def set_default_infer(args):
    # Set Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv("DEVICE_ID", 0)))
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))

def convert_box_xywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    else:
        result = []
        for b in box:
            b = convert_box_xywh_to_xyxy(b)
            result.append(b)               
    return result

def mian():
    parser = get_parser_infer()
    args = parse_args_for_infer(parser)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    set_seed(args.seed)
    set_default_infer(args)
    # Create Network
    network = create_model(
            model_name=args.network.model_name,
            model_cfg=args.network,
            num_classes=args.data.nc,
            sync_bn=False,
            checkpoint_path=args.weight,
        )
    network.set_train(False)
    infer(args, network)


if __name__ == "__main__":
    mian()




