import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import csv

# from models.network_hfpsr import HFPSR as net
from utils import utils_image as util
from utils import utils_option as option
import torch.nn as nn

_result_psnr = list()
_result_ssim = list()
_result_psnr_y = list()
_result_ssim_y = list()


def main(opt, n_model=500000, benchmark='Set5'):
    global _result_psnr, _result_ssim, _result_psnr_y, _result_ssim_y

    model_path = f'superresolution/{opt["task"]}/models/{n_model}_G.pth'
    opt['model_path'] = model_path
    opt['benchmark'] = benchmark
    # opt_test['benchmark'] = benchmark
    # opt_test['folder_lq'] = f'testsets/{benchmark}/LR_bicubic/X{scale}'
    # opt_test['folder_gt'] = f'testsets/{benchmark}/HR'
    training_patch_size = opt['netG']['img_size']

    # args.model_path = f'superresolution/{args.task}/models/{n_model}_G.pth'
    folder_lq = f'testsets/{opt["benchmark"]}/LR_bicubic/X{opt["scale"]}'
    folder_gt = f'testsets/{benchmark}/HR'
    # args.training_patch_size = 48

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
            os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    model = define_model(opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder = f'testsets/{benchmark}/HR'
    save_dir = f'results/{opt["task"]}'
    border = opt['scale']
    window_size = opt['netG']['window_size']

    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(
            opt, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [
                              2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(
            0).to(device)  # CHW-RGB to BCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()

            # if h_old*args.scale != img_gt.shape[0] or w_old*args.scale != img_gt.shape[1]:
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * opt['scale'], :w_old * opt['scale']]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            # CHW-RGB to HWC-BGR
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_{opt["task"]}.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            # float32 to uint8
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            img_gt = img_gt[:h_old * opt['scale'],
                            :w_old * opt['scale'], ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(
                    output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(
                    img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

        _result_psnr.append(ave_psnr)
        _result_psnr_y.append(ave_psnr_y)
        _result_ssim.append(ave_ssim)
        _result_ssim_y.append(ave_ssim_y)
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / \
                len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / \
                len(test_results['ssim_y'])
            print(
                '-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


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
                    cva=opt_net['cva'],
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


def get_image_pair(opt, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'testsets/{opt["benchmark"]}/LR_bicubic/X{opt["scale"]}/{imgname}x{opt["scale"]}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


def test(opt,
         benchmarks=['Set5', 'Set14', 'manga109', 'urban100', 'BSDS100'],
         unit_iter=20000,
         max_iter=500000):
    global _result_psnr, _result_ssim, _result_psnr_y, _result_ssim_y

    for benchmark in benchmarks:
        _iter_list = list()
        for i in range(int(max_iter/unit_iter)):

            n_iter = (i+1)*unit_iter
            _iter_list.append(n_iter)

            main(opt, n_model=n_iter, benchmark=benchmark)

        res = list()
        res.append(_iter_list)
        res.append(_result_psnr)
        res.append(_result_ssim)
        res.append(_result_psnr_y)
        res.append(_result_ssim_y)

        data_to_write = zip(*res)
        save_path = f'benchmark-results/{opt["task"]}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + f'{benchmark}_result.csv', 'w', newline='') as fw:
            wr = csv.writer(fw)
            wr.writerow(('iter', 'psnr', 'ssim', 'psnr_y', 'ssim_y'))
            for data in data_to_write:
                wr.writerow(data)
        _result_psnr = list()
        _result_psnr_y = list()
        _result_ssim = list()
        _result_ssim_y = list()


if __name__ == '__main__':
    test_model_list = ['hfpsr_prototype']
    benchmarks = ['Set5', 'Set14', 'manga109', 'urban100', 'BSDS100']
    unit_iter = 20000
    max_iter = 500000

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=None)
    opt = option.parse(parser.parse_args().opt)

    if opt is not None:
        test(opt, benchmarks, unit_iter, max_iter)
    else:
        for test_model in test_model_list:
            opt = option.parse(f'options/{test_model}.json')
            test(opt, benchmarks, unit_iter, max_iter)
