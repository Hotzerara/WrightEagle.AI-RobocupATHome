import rclpy
from rclpy.node import Node
import torch
from grasp_srv_interface.srv import GroundedSam
import sys
import os
import cv2
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import datetime


# 设置groundedsam地址
GSamDIR = '/root/host_home/ros2_ws/Grounded-SAM-2'
sys.path.append(os.path.join(GSamDIR))

from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

#注册参数值
SAM2_CHECKPOINT = "/root/host_home/ros2_ws/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2.1/sam2.1_hiera_l"
SAM2_CONFIG_DIR = os.path.join(GSamDIR, "sam2/configs")
GROUNDING_DINO_CONFIG = "/root/host_home/ros2_ws/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/root/host_home/ros2_ws/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# create output directory
OUTPUT_DIR = Path("/root/host_home/ros2_ws/docs/data_gsamed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class GroundedSamServer(Node):
    def __init__(self):
        super().__init__('grounded_sam_server')
        self.get_logger().info('Grounded_Sam 服务节点启动中')
        #创建srv服务
        self.srv = self.create_service(
            GroundedSam,
            'grounded_sam_service',
            self.handle_gsam_request
        )
        self.get_logger().info('Grounded_Sam 服务已启动，等待请求...')
        #加载计算设备
        self.get_logger().info(f'计算设备: {DEVICE}')
        #groundedsam模型加载
        self.get_logger().info('Grounded_Sam 模型加载中...')
        self.grounded_sam_model_load()
        self.get_logger().info('Grounded_Sam 模型加载完成.')

    #groundedsam模型加载函数
    def grounded_sam_model_load(self):
        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.get_logger().info(f"正在从{SAM2_MODEL_CONFIG}加载sam2配置文件")
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, sam2_checkpoint, device=DEVICE, config_dir=SAM2_CONFIG_DIR)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

    def get_all_images_from_dir(self, dir_path):
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        dir_path = Path(dir_path)
        image_paths = [str(p) for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in extensions]
        image_paths.sort()
        return image_paths

    def process_single_image(self,image_path,text_prompt,mode):
        """
        对路径列表中图片逐一进行分割
        Args:
            images_path:list 处理后的文件路径列表
            text_prompt:string 分割提示词
            mode 模式选择 0-生成黑白掩码用于graspnet  1-切割物体用于vggt
        """
        self.get_logger().info(f"正在分割图片{image_path}")
        text_prompt = text_prompt + '.'
        image_source, image = load_image(image_path)
        self.sam2_predictor.set_image(image_source)
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # 将掩码数据存入 Detections 对象，方便处理
        detections = sv.Detections(xyxy=input_boxes,mask=masks.astype(bool))

        # 创建白色背景
        white_background = np.full(image_source.shape, 255, dtype=np.uint8)

        # 合并所有掩码
        if detections.mask is not None and len(detections.mask) > 0:
            combined_mask = np.any(detections.mask, axis=0)
            if mode == 0:
                mask_image = (combined_mask * 255).astype(np.uint8)
                return mask_image
            elif mode == 1:
                # 将原图中的物体部分复制到白色背景上
                white_background[combined_mask] = image_source[combined_mask]
                return cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
                # return cv2.cvtColor(white_background, cv2.COLOR_RGB2BGR)


    def handle_gsam_request(self,request,response):
        self.get_logger().info('收到Grounded_Sam请求，正在处理...')
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot(已在process_single_image自动实现无需额外增加)
        text_prompt = request.text_prompt.lower().strip()
        input_path = request.input_path
        mode = request.mode 
        if mode == 0:
            target_filename = "color.png"
            image_path = os.path.join(input_path,target_filename)
            self.get_logger().info(f"成功读取图像: {image_path}")
            mask_image = self.process_single_image(image_path,text_prompt,mode)
            mask_filename = "workspace_mask.png"
            mask_path = os.path.join(input_path,mask_filename)
            cv2.imwrite(str(mask_path), mask_image)
            self.get_logger().info(f"已保存结果至{mask_path}")
            response.success = True
            response.output_path = str(input_path)
            return response
        
        elif mode == 1:
            # 创建各次调用唯一的输出目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            uni_output_dir = OUTPUT_DIR/f"{timestamp}"
            uni_output_dir.mkdir(parents=True,exist_ok=True)
            self.get_logger().info(f"GroundedSam结果将保存至{uni_output_dir}")

            # 输入目录中获取所有图片路径
            images_path = self.get_all_images_from_dir(input_path)
            if not images_path:
                FileExistsError(f"在目录{input_path}中未找到任何符合条件的图片")
            self.get_logger().info(f"找到{len(images_path)}张图片,开始逐一处理...")

            # 开始逐一分割图片
            for image_path in images_path:
                final_image = self.process_single_image(image_path,text_prompt,mode)
                if final_image is not None:
                    output_filename = Path(image_path).name
                    output_path = uni_output_dir/output_filename
                    cv2.imwrite(str(output_path),final_image)
                    self.get_logger().info(f"已保存结果至{output_path}")
            response.success = True
            response.output_path = str(uni_output_dir)
            return response
            

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    config_dir=None,
    **kwargs,
):
    if config_dir is None:
        raise ValueError("config_dir must be provided(absolute path to configs directory)")
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir = config_dir, version_base=None):
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
        OmegaConf.resolve(cfg)
    # Read config and init model
        model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
        
def main():
    rclpy.init()
    grounded_sam_server = GroundedSamServer()
    try:
        rclpy.spin(grounded_sam_server)
    except KeyboardInterrupt:
        grounded_sam_server.get_logger().info('服务器节点被用户手动关闭...')
    finally:
        grounded_sam_server.get_logger().info('服务器节点正在关闭...') 
        grounded_sam_server.destroy_node()
        rclpy.shutdown()
        print('服务器节点已关闭.')