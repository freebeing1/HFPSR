import argparse
from re import L
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os

from models.network_hfpsr import HFPSR
from utils import utils_option as option
from utils.utils_rollout import VITAttentionRollout


class Heatmap:
    def __init__(self, mask_hb, mask_sb, file_name, image_name, save_path='heatmaps', discard_ratio=0.9, head_fusion='mean'):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        self.save_name_hb = os.path.join(save_path, f'heatmap_{discard_ratio}_{head_fusion}_{file_name}_{image_name}_hb.png')
        self.save_name_sb = os.path.join(save_path, f'heatmap_{discard_ratio}_{head_fusion}_{file_name}_{image_name}_sb.png')

        self.mask_hb = cv2.applyColorMap(np.uint8(255 * mask_hb), cv2.COLORMAP_JET)
        self.mask_sb = cv2.applyColorMap(np.uint8(255 * mask_sb), cv2.COLORMAP_JET)

    def save_mask(self):
        cv2.imwrite(self.save_name_hb, self.mask_hb)
        cv2.imwrite(self.save_name_sb, self.mask_sb)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='testsets/Set5/LR_bicubic/X4/babyx4.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--opt', type=str,
                        default='options/hfpsr_ff000001_none_a0_b1_g1_d1.json')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


def show_mask_on_image(img, heatmap):
    img = np.float32(img) / 255
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # cv2.imwrite('heatmap.png', heatmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def define_model(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    if net_type == 'hfpsr':
        from models.network_hfpsr import HFPSR as net
        model = net(upscale=opt_net['upscale'],
                    in_chans=opt_net['in_chans'],
                    img_size=opt_net['img_size'],
                    window_size=opt_net['window_size'],
                    img_range=opt_net['img_range'],
                    ffc_depths=opt_net['ffc_depths'],
                    depths=opt_net['depths'],
                    num_heads=opt_net['num_heads'],
                    ff_layer=opt_net['ff_layer'],
                    embed_dim=opt_net['embed_dim'],
                    mlp_ratio=opt_net['mlp_ratio'],
                    upsampler=opt_net['upsampler'],
                    resi_connection=opt_net['resi_connection'],
                    last_connection=opt_net['last_connection'])
        param_key_g = 'params'
    else:
        raise NotImplementedError(f'net_type [{net_type}] is not found')

    pretrained_model = torch.load(opt['model_path'])
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys(
    ) else pretrained_model, strict=True)

    return model


if __name__ == '__main__':
    args = get_args()

    opt_jsonfile = args.opt.split('/')[-1]
    file_name, file_ext = os.path.splitext(opt_jsonfile)

    dataset_path, benchmarkset, resolution, scale, imagefile = args.image_path.split('/')

    image_name, image_ext = os.path.splitext(imagefile)

    opt = option.parse(args.opt)

    opt['model_path'] = f'superresolution/{file_name}/models/500000_G.pth'
    window_size = opt['netG']['window_size']

    print('# Define Model')
    model = define_model(opt)
    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR).astype(
        np.float32) / 255.
    img = np.transpose(img if img.shape[2] == 1 else img[:, :, [
        2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB

    # print(f'###')
    # print(f'### img.shape: {img.shape}')
    # print(f'###')

    input_tensor = torch.from_numpy(img).float().unsqueeze(
        0).to('cuda')  # CHW-RGB to BCHW-RGB

    with torch.no_grad():
        _, _, h_old, w_old = input_tensor.size()

        # if h_old*args.scale != img_gt.shape[0] or w_old*args.scale != img_gt.shape[1]:
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [2])], 2)[
            :, :, :h_old + h_pad, :]
        input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [3])], 3)[
            :, :, :, :w_old + w_pad]

    # print(f'###')
    # print(f'### input_tensor.shape: {input_tensor.shape}')
    # print(f'###')

    print("# Doing Attention Rollout")
    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                                            discard_ratio=args.discard_ratio)
    mask_hb, mask_sb = attention_rollout(input_tensor)

    heatmap = Heatmap(mask_hb=mask_hb, mask_sb=mask_sb, file_name=file_name, image_name=image_name)
    heatmap.save_mask()

    # name_hb = "attention_rollout_{:.3f}_{}_{}_{}_hb.png".format(
    #     args.discard_ratio, args.head_fusion, file_name, image_name)
    # name_sb = "attention_rollout_{:.3f}_{}_{}_{}_sb.png".format(
    #     args.discard_ratio, args.head_fusion, file_name, image_name)

    # np_img = input_tensor.detach().cpu().numpy()
    # np_img = np.squeeze(np_img)
    # np_img = np.transpose(np_img[[2, 1, 0], :, :], (1, 2, 0))
    # np_img = np.uint8(np_img*255)
    # print(f'###')
    # print(f'### np_img.shape: {np_img.shape}')
    # print(f'###')

    # np_img = cv2.imread('testsets/Set5/HR/baby.png')

    # width_patch = int(mask.shape[0]**0.5)
    # mask = mask.reshape(width_patch, width_patch, mask.shape[1], mask.shape[2])
    # mask = np.transpose(mask, (0, 2, 1, 3))
    # mask = mask.reshape(width_patch*mask.shape[1], -1)
    # temp = np.empty_like(mask)
    # n_patch = mask.shape[0]
    # for i_patch in range(n_patch):
    #     mask[i_patch, :]
    # mask = mask.reshape(np_img.shape[1], np_img.shape[1])
    # cv2.imwrite('mask.png', mask)

    # mask_hb = cv2.resize(mask_hb, (np_img.shape[1], np_img.shape[0]))
    # mask_sb = cv2.resize(mask_sb, (np_img.shape[1], np_img.shape[0]))

    # heatmap_save_path = 'heatmaps'
    # heatmap_hb_save_name = os.path.join(heatmap_save_path, f'heatmap_{args.discard_ratio}_{args.head_fusion}_{file_name}_{image_name}_hb.png')
    # heatmap_sb_save_name = os.path.join(heatmap_save_path, f'heatmap_{args.discard_ratio}_{args.head_fusion}_{file_name}_{image_name}_sb.png')

    # heatmap_hb = cv2.applyColorMap(np.uint8(255 * mask_hb), cv2.COLORMAP_JET)
    # cv2.imwrite(heatmap_hb_save_name, heatmap_hb)
    # heatmap_sb = cv2.applyColorMap(np.uint8(255 * mask_sb), cv2.COLORMAP_JET)
    # cv2.imwrite(heatmap_sb_save_name, heatmap_sb)

    # np_img = np_img[:heatmap_hb.shape[0], :heatmap_hb.shape[1], :]
    # cv2.imwrite("input.png", np_img)
    # masked_hb = show_mask_on_image(np_img, heatmap_hb)
    # masked_sb = show_mask_on_image(np_img, heatmap_sb)
    # # cv2.imwrite("input.png", np_img)
    # cv2.imwrite(name_hb, masked_hb)
    # cv2.imwrite(name_sb, masked_sb)
