## split subject randomly for training and testing
import os
import utils.sortName as getname
from random import randint, sample

# get all subject id: total 48 subjects
# set path
root_path = "/home/gxu/proj1/smatch/data/MRliver"
subj_path = os.path.join(root_path, "DICOM")
slice_path = os.path.join(root_path, "slices")

# select 30 subjects for training and the other 12 subjects for testing, 6 for validation
# get all 48 subjects id
subj_name = getname.get_filename(subj_path, pattern="*.mat")
subj_id = ["subj"+os.path.basename(name).split(".mat")[0] for name in subj_name]
print(len(subj_id))

# get test id, and val id
test_id_obj = open(os.path.join(root_path, "test.list"),'r')
test_id = test_id_obj.read().split("\n")[:12]

val_id_obj = open(os.path.join(root_path, "val.list"),'r')
val_id = val_id_obj.read().split("\n")[:6]

train_val_id = [num for num in subj_id if num not in test_id]
train_id = [num for num in train_val_id if num not in val_id]


# filter all training slices according to selected training subject id
# and filter all other testing slices according to selected testing subjects ID
# 1. get all slice names
slice_names = getname.get_filename(slice_path, pattern="*.h5")

# 2. save all training slices name
with open(os.path.join(root_path, "train_slices.list"),'w') as f:
    # for training slices
    cnt = 0
    for name in train_id:
        for slice_path in slice_names:
            if name in slice_path:
                # save slice name for training
                sel_slice = slice_path.split("slices/")[1]
                f.write(sel_slice[:-3] + '\n')
                cnt += 1                
print(cnt)
# 3. save all testing and validation slices name
with open(os.path.join(root_path, "test_slices.list"),'w') as f:
    # for training slices
    cnt = 0
    for name in test_id:
        for slice_path in slice_names:
            if name in slice_path:
                # save slice name for training
                sel_slice = slice_path.split("slices/")[1]
                f.write(sel_slice[:-3] + '\n')
                cnt += 1                
print(cnt)

with open(os.path.join(root_path, "val_slices.list"),'w') as f:
    # for val slices
    cnt = 0
    for name in val_id:
        for slice_path in slice_names:
            if name in slice_path:
                # save slice name for training
                sel_slice = slice_path.split("slices/")[1]
                f.write(sel_slice[:-3] + '\n')
                cnt += 1                
print(cnt)

# # 4. save all val volume name
# with open(os.path.join(root_path, "val.list"),'w') as f:
#     for sel_slice in val_id:
#         f.write("subj"+sel_slice + '\n')

# # 5. save all testing volume name
# with open(os.path.join(root_path, "test.list"),'w') as f:
#     for sel_slice in test_id:
#         f.write("subj"+sel_slice + '\n')

# 6. save all training volume name
with open(os.path.join(root_path, "train.list"),'w') as f:
    for sel_slice in train_id:
        f.write(sel_slice + '\n')

    
    

