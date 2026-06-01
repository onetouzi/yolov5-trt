import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    """
    解析命令行参数，获取输入模型路径、输出文件路径以及模型任务类型。
    """
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    
    # 必需的输入参数：指定要转换的 PyTorch 权重文件 (.pt) 路径
    parser.add_argument('-w', '--weights', required=True,
                        help='Input weights (.pt) file path (required)')
    
    # 可选的输出参数：指定生成的 .wts 文件保存路径
    parser.add_argument('-o', '--output', 
                        help='Output (.wts) file path (optional)')
    
    # 模型类型参数：默认为目标检测(detect)，也可选分类(cls)或分割(seg)
    parser.add_argument('-t', '--type', type=str, default='detect', 
                        choices=['detect', 'cls', 'seg'],
                        help='determines the model is detection/classification')
    
    args = parser.parse_args()
    
    # 校验输入的权重文件是否存在，不存在则直接报错退出
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    
    # 智能处理输出路径：如果未指定输出路径，默认在原目录下生成同名 .wts 文件
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    # 如果指定的是一个文件夹，则将原文件名拼接到该文件夹下并加上 .wts 后缀
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
            
    return args.weights, args.output, args.type


# ==================== 主程序执行流程 ====================

# 1. 解析命令行传入的参数
pt_file, wts_file, m_type = parse_args()
print(f'Generating .wts for {m_type} model')

# 2. 加载 PyTorch 模型权重
print(f'Loading {pt_file}')
device = select_device('cpu')  # 强制在 CPU 上进行权重提取和格式转换
# 读取 .pt 文件（weights_only=False 允许加载包含复杂对象的字典）
model = torch.load(pt_file, map_location=device, weights_only=False)  
# 【核心防御性加载】优先提取 EMA（指数移动平均）权重以获得更稳定的推理效果；若无则退而使用普通模型权重，并转为 FP32 单精度浮点数
model = model['ema' if model.get('ema') else 'model'].float()

# 3. 针对检测和分割任务的特殊处理：固化 Anchor 尺寸信息
if m_type in ['detect', 'seg']:
    # 将相对大小的 Anchor 乘以对应的 Stride，还原为真实图像上的绝对像素尺寸
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    
    # 删除旧的动态属性，防止导出时发生冲突或数据丢失
    delattr(model.model[-1], 'anchor_grid')  
    
    # 【关键步骤】通过 register_buffer 将计算好的绝对 Anchor 尺寸和 Strides 注册到模型中。
    # 这相当于给它们打上“不可丢弃”的官方标签，确保它们会被写入 state_dict 并随模型一起被导出
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    model.model[-1].register_buffer("strides", model.model[-1].stride)

# 4. 将模型切换为评估模式（关闭 Dropout/BatchNorm 的训练行为）
model.to(device).eval()

# 5. 遍历模型参数，将其序列化写入 .wts 文本文件
print(f'Writing into {wts_file}')
with open(wts_file, 'w') as f:
    # 第一行：写入模型状态字典中的总层数/参数块数量
    f.write('{}\n'.format(len(model.state_dict().keys())))
    
    # 逐层遍历模型的名称(k)和张量(v)
    for k, v in model.state_dict().items():
        # 将多维张量拉平为一维数组，移至 CPU 并转为 NumPy 格式
        vr = v.reshape(-1).cpu().numpy()
        
        # 写入当前层的名称和数据个数
        f.write('{} {} '.format(k, len(vr)))
        
        # 将每一个浮点数值转换为底层字节流，再转成十六进制字符串写入文件
        # '>f' 表示大端模式的单精度浮点数，这是 C++ TensorRT 能够识别的格式
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
            
        # 每一层的数据写完后换行，准备写下一层
        f.write('\n')