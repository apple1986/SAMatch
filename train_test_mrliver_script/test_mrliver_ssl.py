# test for ssl:, like fixmatch
import argparse
import os
import shutil

import h5py
# import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/MRliver', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MRliver/Unimatch_medsam_F2_iter30000_bbox30_lr_1e_2_5e_4_1_labeled_bs4', help='experiment_name')
parser.add_argument('--norm', type=str,
                    default=True, help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_drop', help='model_name: unet, unet_drop')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument("--batch_size", type=int, default=8, help="batch_size used in training")


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # if pred.sum() > 0:
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    # else: 
    #     dice = metric.binary.dc(pred, gt)
    #     hd95 = 0
    #     asd = 0       
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    # image = (image - image.min()) / (image.max()- image.min()+1e-9) # Notice: normalize
    if FLAGS.norm:
        image = (image - image.min()) / (image.max()- image.min()+1e-9) # Notice: normalize
    prediction = np.zeros_like(label)
    for ind in range(0, image.shape[2]):
        slice = image[:, :, ind]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        # slice = (slice - slice.min()) / (slice.max()- slice.min()+1e-9) # Notice: normalize
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[:, :, ind] = pred
        

    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric#, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    # snapshot_path = "./checkpoint/{}_{}_labeled_bs{}/{}".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size, FLAGS.model)
    # test_save_path = "./checkpoint/{}_{}_labeled_bs{}/{}_predictions/".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size, FLAGS.model)
    snapshot_path = "./checkpoint/{}/{}".format(
        FLAGS.exp, FLAGS.model)
    test_save_path = "./checkpoint/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    # save_mode_path = os.path.join(
    #     snapshot_path, 'latest.pth')
    if "FixMatch_standard" in save_mode_path:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True)["state_dict"])
    elif "FixMatch_sam" in save_mode_path:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True)["state_dict"])
    elif "Unimatch" in save_mode_path:
        net.load_state_dict(torch.load(save_mode_path, weights_only=False)["model"])
    else:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True)["state_dict"])
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0

    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        print(f"case: {case}, first_metric: {first_metric}")
        first_total += np.asarray(first_metric)
    avg_metric = first_total / len(image_list)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print("%.4f %.3f %.3f" % (metric[0], metric[1], metric[2]))
    # save_res_path = "./checkpoint/{}_{}_labeled_bs{}/test_metrics.txt".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size)
    save_res_path = "./checkpoint/{}/test_metrics.txt".format(FLAGS.exp)
    with open(save_res_path,'w') as f:
        res = "%.4f\t%.3f\t%.3f" % (metric[0], metric[1], metric[2])
        f.write(res)
