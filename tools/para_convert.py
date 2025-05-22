import mindspore
import torch
from ultralytics import YOLO
from mindyolo.models.model_factory import create_model
from mindyolo.utils.config import load_config, Config
from mindspore.communication import init
init()
def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {    
                    "gamma": "weight",
                    "beta": "bias",
                    "moving_mean": "running_mean",
                    "moving_variance": "running_var",
                    "conv1": "cv1",
                    "conv2": "cv2",
                    "weight": "weight"
                    }
    new_params_list = []
    for ms_param in ms_params.keys():
        without_model_ms_param = ms_param.replace('model.', '', 1)
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param and ".m." not in ms_param and "conv1" not in ms_param and "conv2" not in ms_param:
            ms_param_item = without_model_ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
                print(ms_param, '√')
            else:
                print('pt_param', pt_param)
                print(ms_param, "X = 1")
                break

        elif ".m." in ms_param:
            ms_param_item = without_model_ms_param.split(".")
            pt_param_item = ms_param_item[:-3] + [bn_ms2pt[ms_param_item[-3]]] + [ms_param_item[-2]] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
                print(ms_param, '√')
            else:
                print('pt_param', pt_param)
                print(ms_param, "X = 2")
                break

        elif ("conv1" in ms_param or "conv2" in ms_param) and ".m." not in ms_param:
            ms_param_item = without_model_ms_param.split(".")
            pt_param_item = ms_param_item[:-3] + [bn_ms2pt[ms_param_item[-3]]] + [ms_param_item[-2]] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
                print(ms_param, '√')
            else:
                print('pt_param', pt_param)
                print(ms_param, "X = 3")
                break

        elif "stride" in ms_param:
            ms_value = ms_params[ms_param]
            new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
            print(ms_param, '√')
        # 其他参数
        elif "stride" not in ms_param:
            # 如找到参数对应且shape一致，加入到参数列表
            if without_model_ms_param in pt_params and pt_params[without_model_ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[without_model_ms_param]
                new_params_list.append({"name": ms_param, "data": mindspore.Tensor(ms_value)})
                print(ms_param, '√')
            else:
                print(ms_param, "X = 4")
                break
        
        else:
            print(ms_param, "X = 5")
            break
    # 保存成MindSpore的checkpoint
    mindspore.save_checkpoint(new_params_list, ckpt_path)


def pytorch_params(pth_file):
    # ckpt = torch.load(pth_file, map_location='cpu')
    model = YOLO("/home/ma-user/work/mindyolo/weight/FastSAM-s.pt")
    state_dict = model.model.state_dict()
    # 遍历状态字典中的键和值
    pt_params = {}
    for key, value in state_dict.items():
        # print(value.numpy().shape)
        # f.write(str(model))
        # # 获取键和值的shape
        # shape = value.shape
        pt_params[key] = value.numpy()
    return pt_params

def mindspore_params(ckpt_file):
    cfg, _, _ = load_config(ckpt_file)
    cfg = Config(cfg)
    # print(cfg)
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.network.nc,
        sync_bn=cfg.sync_bn if hasattr(cfg, "sync_bn") else False,
    )
    ms_params = {}
    # f.write(str(network))
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        # f.write(f"{name}---{str(value.shape)}\n")
        ms_params[name] = value
    return ms_params

pth_file = '/home/ma-user/work/mindyolo/weight/FastSAM-s.pt'
ckpt_file = '/home/ma-user/work/mindyolo/configs/fastsam/fastsam-s-native.yaml'
ms_params = mindspore_params(ckpt_file)
print('获得ms参数表')
pt_params = pytorch_params(pth_file)
print('获得torch参数表')
ckpt_path = "/home/ma-user/work/mindyolo/weight/pth2ckpt-fastsams.ckpt"
param_convert(ms_params, pt_params, ckpt_path)
print(f'保存到了：{ckpt_path}')