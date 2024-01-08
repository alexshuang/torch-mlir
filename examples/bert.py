import numpy as np
import torch
import torch_mlir

from transformers import BertConfig, BertForMaskedLM, BertTokenizer

# Wrap the bert model to avoid multiple returns problem
class BertTinyWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = BertConfig.from_json_file("./bert_tiny.json")
        self.bert = BertForMaskedLM(config)
    
    def forward(self, data):
        return self.bert(data)[0]

torch.manual_seed(42)
model = BertTinyWrapper()
model.eval()
data = torch.randint(30522, (2, 4))
np.save('input_tokens.npy', data.numpy())
out_mlir_path = "bert.mlir"

print(model(data))

module = torch_mlir.compile(model, data, output_type="linalg-on-tensors", use_tracing=True)
with open(out_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))

print(f"MLIR IR of tiny bert successfully written into {out_mlir_path}")
