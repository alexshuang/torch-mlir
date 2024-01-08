# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

from PIL import Image
import requests

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


def load_and_preprocess_image(file_path: str):
    img = Image.open(file_path).convert("RGB")
    
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


image_url = "./YellowLabradorLooking_new.jpg"
print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
img_file = 'input_image.npy'
print("save image to " + img_file, file=sys.stderr)
np.save(img_file, img.numpy())

target_models = {
        "resnet18": models.resnet18,
        "mobilenet_v2": models.mobilenet_v2,

        # optional
        #"mobilenet_v3": models.mobilenet_v3_small,
        #"MNASNet": models.mnasnet0_5,
        #"regnet": models.regnet_y_400mf,
        #"resnext": models.resnext50_32x4d,
        #"shufflenet": models.shufflenet_v2_x0_5,
        #"regnet": models.regnet_y_400mf,
        #"squeezenet": models.squeezenet1_0,

        # convert failed
        #"efficientnet_v2": models.efficientnet_v2_s
        #"efficientnet": models.efficientnet_b0
        #"swin_v2_t": models.swin_v2_t
        #"densenet": models.densenet121
        #"convnext": models.convnext_tiny
        #"inception_v3": models.inception_v3
        #"vit": models.vit_b_16
}

for k, model in target_models.items():
    print(f"processing {k} ...")
    m = model(pretrained=True)
    #m.train(False)
    m.eval()

    print(m(img))

    print("torch mlir compiling ...")
    module = torch_mlir.compile(m, img, output_type="linalg-on-tensors")

    mlir = module.operation.get_asm()
    out_mlir_path = f"{k}.mlir"
    open(out_mlir_path, 'w').write(mlir)
    print(f"MLIR IR of {k} successfully written into {out_mlir_path}")

