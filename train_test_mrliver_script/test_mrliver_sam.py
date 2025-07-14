import argparse
import os
import shutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import h5py
# import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import torch.nn.functional as F

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/MRliver', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MRliver/FixMatch_sam_pseudo_label', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=9,
                    help='labeled data')
parser.add_argument("--batch_size", type=int, default=4, help="batch_size used in training")


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, predictor, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    for ind in range(0, image.shape[2]):
        slice = image[:, :, ind]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                # from ssl
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            ## obtain the largesting boundary box

            ## get the highest propability pixel position
            idx = torch.argmax(out_main[0,1,:,:])
            r_idx, c_idx = idx // 256, idx % 256
            # input_points = torch.tensor([[int(c_idx),int(r_idx)]]).cuda()
            input_points = torch.tensor([[45, 150], [43, 150], [10,10]]).cuda() * 4
            input_box = torch.tensor([[26, 134, 60, 165]]).cuda() * 4
            ## from sam
            img_sam = F.interpolate(input, size=(1024,1024)) #BxCx1024x1024
            img_sam = torch.cat((img_sam, img_sam, img_sam), dim=1) # B3HW
            # img_sam = (img_sam - img_sam.min()) / (img_sam.max() - img_sam.min())
            predictor.set_torch_image(img_sam, out.shape)
            mask, _, _ = predictor.predict_torch(
                                            point_coords = input_points.unsqueeze(0),
                                            point_labels=torch.tensor([[1,1, 0]]).cuda(),
                                            # mask_input=out.float().unsqueeze(0),
                                            boxes = input_box,
                                            multimask_output=False,)
             
            # img_sam = np.stack((slice, slice, slice), axis=2)
            # predictor.set_image(img_sam.astype("uint8"))
            # mask, _, _ = predictor.predict(
            #                 mask_input=out[None, :, :],
            #                 multimask_output=False,
            #             ) 
            # plt.imshow(mask); plt.show()

            mask = mask.cpu().detach().long().numpy()[0,0,:,:] # HW
            plt.imshow(mask); plt.show()
            out = out.cpu().detach().numpy()
            plt.imshow(out); plt.show()
            # plt.imshow(label[:, :, ind]); plt.show()
            pred = zoom(mask, (x / 256, y / 256), order=0)
            prediction[:, :, ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric#, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "./checkpoint/{}_{}_labeled_bs{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size, FLAGS.model)
    test_save_path = "./checkpoint/{}_{}_labeled_bs{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    if "FixMatch_standard" in save_mode_path:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True)["state_dict"])
    elif "FixMatch_sam" in save_mode_path:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True)["state_dict"])
    else:
        net.load_state_dict(torch.load(save_mode_path, weights_only=True))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0

    # build sam
    sam = sam_model_registry["vit_b"](checkpoint="./checkpoint/SAM/sam_vit_b_01ec64.pth").cuda()
    predictor = SamPredictor(sam)
    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, net, predictor, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
    avg_metric = first_total / len(image_list)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print("%.4f %.3f %.3f" % (metric[0], metric[2], metric[1]))
    save_res_path = "./checkpoint/{}_{}_labeled_bs{}/test_metrics.txt".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.batch_size)
    with open(save_res_path,'w') as f:
        res = "%.4f %.3f %.3f" % (metric[0], metric[2], metric[1])
        f.write(res)
