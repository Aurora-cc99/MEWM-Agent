"""Data augmentation utilities for MEFlowNet training."""




        











        






        








        


























import io
import random
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from torchvision.utils import save_image
def to_tensor_uint8(img):
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            raise ValueError("numpy input expected uint8")
        img = Image.fromarray(img)
    t = TF.to_tensor(img)
    return t

def to_pil(img_tensor):
    return TF.to_pil_image(img_tensor.clamp(0,1))

def ensure_square_crop_bbox(x0, y0, x1, y1, img_w, img_h, square_expand=True):
    w = x1 - x0
    h = y1 - y0
    if square_expand:
        s = max(w, h)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0_new = cx - s/2.0
        x1_new = cx + s/2.0
        y0_new = cy - s/2.0
        y1_new = cy + s/2.0
    else:
        x0_new, y0_new, x1_new, y1_new = x0, y0, x1, y1
    x0c = max(0, math.floor(x0_new))
    y0c = max(0, math.floor(y0_new))
    x1c = min(img_w, math.ceil(x1_new))
    y1c = min(img_h, math.ceil(y1_new))
    return x0c, y0c, x1c, y1c

def crop_image_tensor(img_tensor, x0, y0, x1, y1):
    _, H, W = img_tensor.shape
    x0i, y0i = int(x0), int(y0)
    x1i, y1i = int(x1), int(y1)
    return img_tensor[:, y0i:y1i, x0i:x1i].contiguous()

def crop_flow_tensor(flow_tensor, x0, y0, x1, y1):
    _, H, W = flow_tensor.shape
    x0i, y0i = int(x0), int(y0)
    x1i, y1i = int(x1), int(y1)
    return flow_tensor[:, y0i:y1i, x0i:x1i].contiguous()

def crop_landmarks(lm, x0, y0):
    lm2 = lm.copy() if isinstance(lm, np.ndarray) else lm.clone()
    lm2[..., 0] = lm2[..., 0] - x0
    lm2[..., 1] = lm2[..., 1] - y0
    return lm2

def resize_image_tensor(img_tensor, out_h, out_w, interp_mode='bilinear'):
    t = img_tensor.unsqueeze(0)
    t2 = F.interpolate(t, size=(out_h, out_w), mode=interp_mode, align_corners=False if interp_mode=='bilinear' else None)
    return t2.squeeze(0)

def resize_flow_tensor(flow, out_h, out_w):
    _, H, W = flow.shape
    f = flow.unsqueeze(0)
    f_res = F.interpolate(f, size=(out_h, out_w), mode='bilinear', align_corners=False)
    f_res = f_res.squeeze(0)
    scale_x = float(out_w) / float(W)
    scale_y = float(out_h) / float(H)
    f_res = f_res.clone()
    f_res[0, :, :] = f_res[0, :, :] * scale_x
    f_res[1, :, :] = f_res[1, :, :] * scale_y
    return f_res

def flip_horizontal_image_tensor(img_tensor):
    return torch.flip(img_tensor, dims=[2])

def flip_horizontal_flow(flow):
    f = torch.flip(flow, dims=[2])
    f = f.clone()
    f[0] = -f[0]
    return f

def flip_landmarks_x(lm, img_w):
    lm2 = lm.copy() if isinstance(lm, np.ndarray) else lm.clone()
    lm2[..., 0] = (img_w - 1) - lm2[..., 0]
    return lm2


def random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.15, p=0.8):
    if random.random() > p:
        return img
    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    s = 1.0 + random.uniform(-saturation, saturation)
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    return img

from PIL import ImageFilter

def random_gaussian_blur(img, kernel_max=5, p=0.5):
    if random.random() > p:
        return img
    pil = to_pil(img)
    r = random.uniform(0.1, 1.5)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=r))
    return TF.to_tensor(pil)

def random_noise(img, std_max=0.02, p=0.5):
    if random.random() > p:
        return img
    std = random.uniform(0.0, std_max)
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0,1)

def random_jpeg_compress(img, p=0.3, qmin=60, qmax=95):
    if random.random() > p:
        return img
    pil = to_pil(img)
    buf = io.BytesIO()
    q = random.randint(qmin, qmax)
    pil.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    pil2 = Image.open(buf).convert('RGB')
    return TF.to_tensor(pil2)

def random_occlusion(img, max_h_ratio=0.12, max_w_ratio=0.25, p=0.4):
    if random.random() > p:
        return img
    C, H, W = img.shape
    h = int(random.uniform(0.05, max_h_ratio) * H)
    w = int(random.uniform(0.05, max_w_ratio) * W)
    x0 = random.randint(int(0.05 * W), max(1, W - w - 1))
    y0 = random.randint(int(0.05 * H), max(1, H - h - 1))
    img[:, y0:y0 + h, x0:x0 + w] = torch.rand(C,1,1, device=img.device)
    return img


def preprocess_pair(
    I_n, I_e, lm_n, lm_e, facial_flow, head_flow,
    out_size=384,
    face_expand=1.25,
    augment=True,
    device='cpu'
):
    if isinstance(I_n, torch.Tensor):
        img_n = I_n.clone()
        if img_n.max() > 2.0:
            img_n = img_n.float() / 255.0
    else:
        img_n = to_tensor_uint8(I_n)
    if isinstance(I_e, torch.Tensor):
        img_e = I_e.clone()
        if img_e.max() > 2.0:
            img_e = img_e.float() / 255.0
    else:
        img_e = to_tensor_uint8(I_e)

    if not isinstance(facial_flow, torch.Tensor):
        facial_flow = torch.from_numpy(facial_flow).float()
    else:
        facial_flow = facial_flow.clone().float()

    if not isinstance(head_flow, torch.Tensor):
        head_flow = torch.from_numpy(head_flow).float()
    else:
        head_flow = head_flow.clone().float()

    lm_n_np = lm_n.copy() if isinstance(lm_n, np.ndarray) else lm_n.detach().cpu().numpy()
    lm_e_np = lm_e.copy() if isinstance(lm_e, np.ndarray) else lm_e.detach().cpu().numpy()

    _, H_full, W_full = img_n.shape

    all_lms = np.concatenate([lm_n_np, lm_e_np], axis=0) if lm_e_np is not None else lm_n_np
    x_min = float(np.min(all_lms[:, 0]))
    y_min = float(np.min(all_lms[:, 1]))
    x_max = float(np.max(all_lms[:, 0]))
    y_max = float(np.max(all_lms[:, 1]))

    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    s = max(w, h) * face_expand
    x0 = cx - s/2.0
    y0 = cy - s/2.0
    x1 = cx + s/2.0
    y1 = cy + s/2.0

    x0c, y0c, x1c, y1c = ensure_square_crop_bbox(x0, y0, x1, y1, W_full, H_full, square_expand=False)
    crop_w = x1c - x0c
    crop_h = y1c - y0c

    img_n_crop = crop_image_tensor(img_n, x0c, y0c, x1c, y1c)
    img_e_crop = crop_image_tensor(img_e, x0c, y0c, x1c, y1c)
    facial_flow_crop = crop_flow_tensor(facial_flow, x0c, y0c, x1c, y1c)
    head_flow_crop = crop_flow_tensor(head_flow, x0c, y0c, x1c, y1c)
    lm_n_crop = crop_landmarks(lm_n_np, x0c, y0c)
    lm_e_crop = crop_landmarks(lm_e_np, x0c, y0c)

    if augment:
        max_tx = int(0.08 * crop_w)
        max_ty = int(0.08 * crop_h)
        tx = random.randint(-max_tx, max_tx) if max_tx > 0 else 0
        ty = random.randint(-max_ty, max_ty) if max_ty > 0 else 0
        if tx != 0 or ty != 0:
            C, Hc, Wc = img_n_crop.shape
            x0s = max(0, tx)
            y0s = max(0, ty)
            x1s = x0s + Wc - abs(tx)
            y1s = y0s + Hc - abs(ty)
            if 0 <= x0s < x1s <= Wc and 0 <= y0s < y1s <= Hc:
                img_n_crop = img_n_crop[:, y0s:y1s, x0s:x1s]
                img_e_crop = img_e_crop[:, y0s:y1s, x0s:x1s]
                facial_flow_crop = facial_flow_crop[:, y0s:y1s, x0s:x1s]
                head_flow_crop = head_flow_crop[:, y0s:y1s, x0s:x1s]
                lm_n_crop[..., 0] = lm_n_crop[..., 0] - x0s
                lm_n_crop[..., 1] = lm_n_crop[..., 1] - y0s
                lm_e_crop[..., 0] = lm_e_crop[..., 0] - x0s
                lm_e_crop[..., 1] = lm_e_crop[..., 1] - y0s
                crop_h = y1s - y0s
                crop_w = x1s - x0s

    img_n_rs = resize_image_tensor(img_n_crop, out_size, out_size, interp_mode='bilinear')
    img_e_rs = resize_image_tensor(img_e_crop, out_size, out_size, interp_mode='bilinear')
    facial_flow_rs = resize_flow_tensor(facial_flow_crop, out_size, out_size)
    head_flow_rs = resize_flow_tensor(head_flow_crop, out_size, out_size)
    scale_x = out_size / float(crop_w)
    scale_y = out_size / float(crop_h)
    lm_n_rs = lm_n_crop.astype(np.float32)
    lm_e_rs = lm_e_crop.astype(np.float32)
    lm_n_rs[..., 0] = lm_n_rs[..., 0] * scale_x
    lm_n_rs[..., 1] = lm_n_rs[..., 1] * scale_y
    lm_e_rs[..., 0] = lm_e_rs[..., 0] * scale_x
    lm_e_rs[..., 1] = lm_e_rs[..., 1] * scale_y

    if augment and random.random() < 0.5:
        img_n_rs = flip_horizontal_image_tensor(img_n_rs)
        img_e_rs = flip_horizontal_image_tensor(img_e_rs)
        facial_flow_rs = flip_horizontal_flow(facial_flow_rs)
        head_flow_rs = flip_horizontal_flow(head_flow_rs)
        lm_n_rs = flip_landmarks_x(lm_n_rs, out_size)
        lm_e_rs = flip_landmarks_x(lm_e_rs, out_size)

    if augment:
        img_n_rs = random_color_jitter(img_n_rs, p=0.9)
        img_e_rs = random_color_jitter(img_e_rs, p=0.9)
        img_n_rs = random_gaussian_blur(img_n_rs, p=0.3)
        img_e_rs = random_gaussian_blur(img_e_rs, p=0.3)
        img_n_rs = random_noise(img_n_rs, p=0.4)
        img_e_rs = random_noise(img_e_rs, p=0.4)
        img_n_rs = random_jpeg_compress(img_n_rs, p=0.25)
        img_e_rs = random_jpeg_compress(img_e_rs, p=0.25)
        img_n_rs = random_occlusion(img_n_rs, p=0.25)
        img_e_rs = random_occlusion(img_e_rs, p=0.25)

    img_n_rs = img_n_rs.clamp(0.0, 1.0)
    img_e_rs = img_e_rs.clamp(0.0, 1.0)

    lm_n_t = torch.from_numpy(lm_n_rs).float()
    lm_e_t = torch.from_numpy(lm_e_rs).float()
    return img_n_rs.to(device)*255.0, img_e_rs.to(device)*255.0, lm_n_t.to(device), lm_e_t.to(device), facial_flow_rs.to(device), head_flow_rs.to(device)





















