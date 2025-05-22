import torch
import mindspore
from ultralytics import YOLO
from mindyolo.models.model_factory import create_model
from mindyolo.utils.config import load_config, Config
from mindspore.communication import init
init()
# 通过PyTorch参数文件，打印PyTorch所有参数名和shape，返回字典
def pytorch_params(pth_file):
    ckpt = torch.load(pth_file, map_location='cpu')
    model = ckpt['model']
    state_dict = model.state_dict()
    # 遍历状态字典中的键和值
    with open('/home/ma-user/work/weight_txt/FastSAM-s.txt', 'w') as f:
        pt_params = {}
        for key, value in state_dict.items():
            # print(value.numpy().shape)
            # f.write(str(model))
            # # 获取键和值的shape
            # shape = value.shape
            f.write(f"{key}---{str(value.numpy().shape)}\n")
            pt_params[key] = value.numpy()
        return pt_params
    

def mindspore_params():
    cfg, _, _ = load_config('/home/ma-user/work/mindyolo/configs/fastsam/fastsam-s-native.yaml')
    cfg = Config(cfg)
    print(cfg)
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.network.nc,
        sync_bn=cfg.sync_bn if hasattr(cfg, "sync_bn") else False,
    )
    ms_params = {}
    with open('/home/ma-user/work/weight_txt/ms-fastsam-s-native.txt', 'w') as f:
        print(1)
        # f.write(str(network))
        for param in network.get_parameters():
            name = param.name
            value = param.data.asnumpy()
            f.write(f"{name}---{str(value.shape)}\n")
            # print(type(value))
            ms_params[name] = value
    return ms_params

def pytorch_yolov8_seg():
    # model = YOLO("yolov8x-seg.yaml").load('/home/ma-user/work/FastSAM-x.pt')
    model = YOLO("/home/ma-user/work/mindyolo/weight/FastSAM-s.pt")
    print('*'*20)
    print(model.model)
    state_dict = model.model.state_dict()

    with open('/home/ma-user/work/weight_txt/FastSAM-s.txt', 'w') as f:
        pt_params = {}
        for key, value in state_dict.items():
            # print(value.numpy().shape)
            # f.write(str(model))
            # # 获取键和值的shape
            f.write(f"{key}---{str(value.numpy().shape)}\n")
            pt_params[key] = value.numpy()
        return pt_params
    

# pytorch_yolov8_seg()
# pth_path = "/home/ma-user/work/mindyolo/weight/FastSAM-s.pt"
# pytorch_params(pth_path)
# pt_param = pytorch_params(pth_path)
mindspore_params()
# print("="*20)