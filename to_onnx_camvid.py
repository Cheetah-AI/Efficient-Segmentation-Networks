import torch
import torch.onnx as onnx
from model.ENet import ENet

model_name = "enet_1000epochs_camvid"

# Load the PyTorch model
model = ENet(classes=11, onnx_friendly=True)

# Load the state dict from the .pth file
state_dict = torch.load(model_name + '.pth')

# Load state dict into the model
model.load_state_dict(state_dict['model'])

# Get input shape from the model's first layer
# input_shape = list(model.parameters())[0].shape[1:]
input_shape = [1, 3, 360, 480]

# Create a dummy input tensor based on the inferred input shape
dummy_input = torch.randn(*input_shape)

# Forward pass to get the output shape
model.eval()
# with torch.no_grad():
#     output = model(dummy_input)
# output_shape = output.shape[1:]

print("Inferred Input Shape:", input_shape)
# print("Inferred Output Shape:", output_shape)

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Convert to ONNX
onnx_file_path = model_name + ".onnx"
input_names = ["input"]
output_names = ["output"]
onnx.export(traced_model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names, autograd_inlining=False, opset_version=10)

print("Model converted to ONNX successfully!")
