import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import pywt

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/gxu/proj1/smatch', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='checkpoint/ACDC/UNet_Fully_Supervised_140', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice ,hd95, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/ACDC/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            else:
                # coeffs = pywt.dwt2(slice, 'bior1.1')
                # cA, (cH, cV, cD) = coeffs
                # l = cA
                # h = cH+ cV+ cD
                # l = zoom(l, (256 / 128, 256 / 128), order=0)
                # h = zoom(h, (256 / 128, 256 / 128), order=0)
                # # input_1 = torch.from_numpy(l).float()
                # # input_2 = torch.from_numpy(h).float()
                # input_1 = torch.from_numpy(l).unsqueeze(
                #     0).unsqueeze(0).float().cuda()
                # input_2 = torch.from_numpy(h).unsqueeze(
                #     0).unsqueeze(0).float().cuda()
                # out_main,_ = net(input_1,input_2)
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/data/ACDC/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "/home/gxu/proj1/smatch/{}/{}".format( #_labeled
        FLAGS.exp,  FLAGS.model)
    test_save_path = "/home/gxu/proj1/smatch/{}/{}_predictions/".format( #_labeled
        FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path, weights_only=True))#)['model'])#)['state_dict'])
    #net.load_state_dict(torch.load("/home/gxu/proj1/smatch/ACDC/FixMatch_std_SAMUS_F3_case1_3/unet/model_iter_30000.pth")['state_dict'])
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    mean_metric = np.array((metric[0]+metric[1]+metric[2])/3)
    print(metric)
    print(mean_metric)
    save_res_path = "./{}/test_metrics.txt".format(FLAGS.exp)
    with open(save_res_path,'w') as f:
        for n in range(3):
            res = "%.4f\t%.3f\t%.3f\n" % (metric[n][0], metric[n][1], metric[n][2])
            f.write(res)
        res_dice = "Dice RV, MYO and LV:\n%.4f\t%.4f\t%.4f\n" % (metric[0][0], metric[1][0], metric[2][0])
        f.write(res_dice)        
        res = "Mean Dice, HD95 and ASD\n%.4f\t%.3f\t%.3f\n" % (mean_metric[0], mean_metric[1], mean_metric[2])
        f.write(res)