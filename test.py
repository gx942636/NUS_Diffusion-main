import numpy as np
import torch
import model2 as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics
import os
import random
import scipy.io as scio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_256_256_val.json',
                        help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_wandb_ckpt', action='store_true')
parser.add_argument('-log_eval', action='store_true')

args = parser.parse_args(args=[])
opt = Logger.parse(args)
opt = Logger.dict_to_nonedict(opt)

# logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def pad_data(data, block_size):
    # 获取数据的形状
    height, width = data.shape

    # 计算需要填充的行和列数
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size

    # 在数据的底部和右侧填充0
    padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_data, pad_height, pad_width


def block_data(data, block_size):
    # 对数据进行填充
    padded_data, pad_height, pad_width = pad_data(data, block_size)

    # 获取填充后的数据的形状
    height, width = padded_data.shape

    # 计算水平和垂直方向上的块数
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size

    # 创建一个空数组来存储分块后的数据
    blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size), dtype=padded_data.dtype)

    # 分块
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[idx] = block
            idx += 1

    return blocks, padded_data.shape, pad_height, pad_width


def reconstruct_data(blocks, pad_shape, pad_height, pad_width):
    # 获取填充后数据的形状
    height, width = pad_shape

    # 获取块的形状和数量
    num_blocks, block_height, block_width = blocks.shape

    # 计算水平和垂直方向上的块数
    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width

    # 创建一个空数组来存储重构后的数据
    reconstructed_data = np.empty((height, width), dtype=blocks.dtype)

    # 重构数据
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width] = block
            idx += 1

    # 删除填充的部分
    final_data = reconstructed_data[:height if not pad_height else -pad_height, :width if not pad_width else -pad_width]

    return final_data


data_path = f"{opt['datasets']['test']['real_data_path']}.mat"
sample_rate = opt['datasets']['test']['sample_rate']
ori_spec = scio.loadmat(data_path)['spec']
N1 = ori_spec.shape[0]
N2 = ori_spec.shape[1]

threshold = 0.01
mask = np.abs(ori_spec) < np.max(np.abs(ori_spec), axis=1, keepdims=True)[0] * threshold
# 使用掩码将原始张量中对应位置的值置为零
ori_spec[mask] = 0

phase_input, pad_shape, pad_height, pad_width = block_data(ori_spec, 256)
xx = np.fft.ifft(ori_spec, axis=0)

result_formatted = random.randint(1, 1000)  # 生成1到10000之间的随机整数
print(result_formatted)
Mask = scio.loadmat(f"./data/{sample_rate}_stack_mask-{N1}-{N2}/Mask_{result_formatted}.mat")['Mask']
U = Mask

NUS_FID = np.multiply(U, xx)
NOISE_input = np.fft.fft(NUS_FID, axis=0)
NUS_FID, _, _, _ = block_data(NUS_FID, 256)
NOISE_input, _, _, _ = block_data(NOISE_input, 256)

GT_label = np.fft.fft(xx, axis=0)
GT_label, _, _, _ = block_data(GT_label, 256)
block_num = phase_input.shape[0]
print(block_num)

max_amp = np.max(np.abs(NOISE_input))
NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float()) / (
            max_amp * 3.5)
NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / (
    max_amp)
GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / (
            max_amp * 8)

LR = NUS_FID.unsqueeze(1)
SR = torch.cat((NOISE_input.real.unsqueeze(1), NOISE_input.imag.unsqueeze(1)), dim=1)
HR = torch.cat((GT_label.real.unsqueeze(1), GT_label.imag.unsqueeze(1)), dim=1)
print(GT_label.shape)
val_data = {'LR': LR, 'HR': HR, 'SR': SR}

diffusion = Model.create_model(opt)
avg_mse = 0.0
result_path = '{}/{}'.format(opt['path']['results'], 100)
os.makedirs(result_path, exist_ok=True)
diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

val_data = diffusion.set_device(val_data)
diffusion.feed_data(val_data)
diffusion.test(continous=False)
visuals = diffusion.get_current_visuals()

try:
    if block_num is not None:
        fake_img = reconstruct_data(torch.complex(visuals['INF'][0 : block_num , 0], visuals['INF'][0 : block_num , 1]).float().cpu().numpy(), pad_shape, pad_height, pad_width)
        sr_img = reconstruct_data(torch.complex(visuals['SR'][-block_num : , 0], visuals['SR'][-block_num : , 1]).float().cpu().numpy(), pad_shape, pad_height, pad_width)
        sr_spec = reconstruct_data(torch.complex(visuals['SR_Spec'][-block_num : , 0], visuals['SR_Spec'][-block_num :, 1]).float().cpu().numpy(), pad_shape, pad_height, pad_width)
        hr_img = reconstruct_data(torch.complex(visuals['HR'][0 : block_num , 0], visuals['HR'][0 : block_num , 1]).float().cpu().numpy(), pad_shape, pad_height, pad_width)
        lr_img = reconstruct_data(visuals['LR'][0 : block_num].squeeze(1).float().cpu().numpy(), pad_shape, pad_height, pad_width)

except NameError:
    fake_img = torch.complex(visuals['INF'][0, 0], visuals['INF'][0, 1]).squeeze().float().cpu().numpy()
    sr_img = torch.complex(visuals['SR'][-1, 0], visuals['SR'][-1, 1]).squeeze().float().cpu().numpy()
    sr_spec = torch.complex(visuals['SR_Spec'][-1, 0], visuals['SR_Spec'][-1, 1]).squeeze().float().cpu().numpy()
    hr_img = torch.complex(visuals['HR'][0, 0], visuals['HR'][0, 1]).squeeze().float().cpu().numpy()
    lr_img = visuals['LR'][0].squeeze().float().cpu().numpy()

Metrics.save_contour2(
nus_spec=fake_img,
recon_spec=sr_img,
label_spec=hr_img,
sr_spec=sr_spec,  # result of densenet
save_path='{}/{}_{}_hr.png'.format(result_path, 200, 1))
plt.show()