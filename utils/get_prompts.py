import cv2
import numpy as np
import torch
import random



## box
def find_box(mask):
    # find box: the left-upper and right-bottom points
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    input_box = np.array([x, y, x+w, y+h])
    return input_box

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def get_bbox256_torch(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
        bounding box coordinates in the resized image
    """
    B, H, W = mask_256.shape
    bboxes256 = torch.ones((B, 1, 4)).to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = mask_256[n,:,:]
        idx_fg = torch.argwhere(pd_one > 0.5)
        if (idx_fg.sum() > 0):
            # print(idx_fg)
            # idx_bg = torch.argwhere(pd_one < 0.5)
            x_min, x_max = torch.min(idx_fg[:,1]), torch.max(idx_fg[:, 1])
            y_min, y_max = torch.min(idx_fg[:,0]), torch.max(idx_fg[:, 0])
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes256[n, 0, :] = torch.tensor([x_min, y_min, x_max, y_max])
            # print(bboxes256)
            # print("*"*50)
    return bboxes256

def get_bbox256_cv(mask_256, bbox_shift=3):
    B, H, W = mask_256.shape
    binary_mask = mask_256.detach().cpu().numpy()
    bboxes256 = np.ones((B, 1, 4))* (-100)#.to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = binary_mask[n, :, :].astype(np.uint8)
        if (pd_one.sum()> 0):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(pd_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize variables to keep track of the largest bounding box
            max_area = 0
            
            for contour in contours:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Update the largest bounding box if this one is bigger
                if area > max_area:
                    max_area = area
                    x_min = max(0, x - bbox_shift)
                    x_max = min(W, x + w + bbox_shift)
                    y_min = max(0, y - bbox_shift)
                    y_max = min(H, y + h + bbox_shift)
                    bboxes256[n, 0, :] = np.array([x_min, y_min, x_max, y_max])
    bboxes256 = torch.tensor(bboxes256).to(mask_256.device)

    return bboxes256  

##################point
def find_centroid(mask):
    # find the centeroid of the object
    # calculate moments of binary image
    # return xy direction
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    cCol = int(M["m10"] / M["m00"])
    cRow = int(M["m01"] / M["m00"])
    # print(cRow, " ", cCol)
    return np.array([[cCol, cRow]])
################################### find the high confident point
def find_position(unlabel_pd, conf_thresh=0.95):
    B, H, W = unlabel_pd.shape 
    temp_pd = unlabel_pd.view(B, -1)           
    M = temp_pd.argmax(1) # B
    ## if the gt is blank, it will result in error prompt
    is_null = temp_pd.sum(dim=1) < conf_thresh
    if is_null.sum() > 0:
        M[is_null] = 0
        # M[is_null] = torch.tensor(32896, device=M.device)
    #     print("Here")
    # else:
    #     print("is not null")

    idx = torch.cat(((M / H).view(-1, 1), (M % W).view(-1, 1)), dim=1).long()
    idx[:, [0,1]] = idx[:, [1,0]] # (Y,X) --> (X,Y)
    input_points = idx.unsqueeze(1)
    # input_labels = torch.ones((B, 1), device=args.cuda_num)
    input_labels = (~is_null).float().reshape(B, -1) #torch.ones((B, 1), device=args.cuda_num)
    point_label = (input_points, input_labels)
    return point_label

def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], point_label

def generate_click_prompt(img, msk, pt_label = 1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

def generate_unique_random_numbers(n, start, end):
    # generate n number in [start, end]
    return random.sample(range(start, end + 1), n)

def find_point_label(gt_sam):
    # get 10 points for gt
    B, _, _ = gt_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=gt_sam.device)
    points_label = torch.zeros((B, 10), device=gt_sam.device)
    SEL_PT_NUM = 1 # how many positive poinsts are selected?
    for n in range(B):
        gt_one = gt_sam[n,:,:] # HW
        idx_fg = torch.argwhere(gt_one == 1) # 1 is the class label
        idx_bg = torch.argwhere(gt_one != 1)
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(SEL_PT_NUM, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n,:SEL_PT_NUM,:] = idx_fg[random_numbers]
            points_label[n,:SEL_PT_NUM] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(10-SEL_PT_NUM, 0, len(idx_bg)-1)
            points_coord[n,SEL_PT_NUM:,] = idx_bg[random_numbers]
            points_label[n,SEL_PT_NUM:] = 0
    return points_coord, points_label

def find_point_label_pseudo(pd_sam):
    # input: pd_sam is a probability map
    # get 10 points for pseudo-label: 1 for highest probability, 9 for bg
    B, _, _ = pd_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=pd_sam.device)
    points_label = torch.zeros((B, 10), device=pd_sam.device)
    for n in range(B):
        pd_one = pd_sam[n,:,:] # HW
        idx_fg = torch.argwhere(pd_one > 0.5).to(device=pd_sam.device) # 1 is the class label, 0.5 is threshold
        idx_bg = torch.argwhere(pd_one < 0.5).to(device=pd_sam.device) 
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(1, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n, :1, :] = idx_fg[random_numbers]
            points_label[n, :1] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(9, 0, len(idx_bg)-1)
            points_coord[n, 1:, :] = idx_bg[random_numbers]
            points_label[n, 1:] = 0
    return points_coord, points_label