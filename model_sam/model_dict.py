from model_sam.segment_anything.build_sam import sam_model_registry
from model_sam.segment_anything_samus.build_sam_us import samus_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.ckpt)
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint=args.ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
