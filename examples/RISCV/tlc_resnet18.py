from PIL import Image

import torchvision.models as models
from torchvision import transforms
import torch
import torch._lazy
import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend


def load_and_preprocess_image(url: str, device):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    #img = Image.open(requests.get(url, headers=headers,
    #                              stream=True).raw).convert("RGB")

    file_path = "/work/YellowLabradorLooking_new.jpg"
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


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


# Register the example LTC backend.
lazy_backend._initialize()

device = 'lazy'

resnet18 = models.resnet18(pretrained=True)
resnet18.to(device)
resnet18.eval()

img = torch.randn((1, 3, 224, 224), device=device)
logits = resnet18(img)

# Mark end of training/evaluation iteration and lower traced graph.
torch._lazy.mark_step()
print('logits:', logits.shape, logits)

# Optionally dump MLIR graph generated from LTC trace.
computation = lazy_backend.get_latest_computation()
if computation:
    #mlir_raw = computation.to_string()
    #open('torch_resnet18.mlir', 'w').write(mlir_raw)
