from executorch.exir.backend.backend_api import to_backend
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export, ExportedProgram
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import export_for_training
import torch
from torch.export import export, ExportedProgram, Dim
from PIL import Image

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


class SimpleConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=3, groups=3, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, groups=3, kernel_size=3, padding=1, bias=False
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, groups=3, kernel_size=3, padding=1, bias=False
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, groups=3, kernel_size=3, padding=1, bias=False
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, groups=3, kernel_size=3, padding=1, bias=False
        )
        self.relu = torch.nn.ReLU()
        # Set the convolution weights to be a 3x3 Gaussian smoothing kernel
        gaussian_kernel = torch.tensor(
            [[1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]], dtype=torch.float32
        ).repeat(3, 1, 1, 1)  # Repeat for each input channel
        box = torch.tensor(
            [[1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9]], dtype=torch.float32
        ).repeat(3, 1, 1, 1)

        box = gaussian_kernel
        self.conv.weight.data = box
        self.conv2.weight.data = box
        self.conv3.weight.data = box
        self.conv4.weight.data = box
        self.conv5.weight.data = box

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        a = self.conv2(a)
        a = self.relu(a)
        a = self.conv3(a)
        a = self.conv4(a)
        a = self.conv5(a)
        return a


# Load the image
image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")

# Transform the image to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256, 256))  # Resize to match the example_args size
])
image_tensor_sample = transform(image).unsqueeze(0)  # Add batch dimension

# Apply the SimpleConv model
# model = SimpleConv()


def process_image(image_tensor, model):
    output = model(image_tensor)
    print(output)

    # Print the output shape
    print(output.shape)

    # Convert the output tensor to a numpy array
    output_image = output.squeeze(0).permute(1, 2, 0).detach().numpy()

    # Display the image
    plt.imshow(output_image)
    plt.axis('off')  # Turn off axis
    plt.show()


# --------------
# Dynamic shapes
dim2_x = Dim("dim2_x", min=64, max=1024)
dim3_x = Dim("dim3_x", min=64, max=1024)
dynamic_shapes = {"x": {1: 3, 2: dim2_x, 3: dim3_x}}
# example_args = (torch.randn(1, 3, 256, 256),)
example_args = (image_tensor_sample,)
aten_dialect: ExportedProgram = export(
    SimpleConv(), example_args, dynamic_shapes=dynamic_shapes)
print(aten_dialect)


out = aten_dialect.module()(torch.randn(1, 3, 251, 254))
print(out.shape)


edge: EdgeProgramManager = to_edge_transform_and_lower(
    aten_dialect,
    partitioner=[XnnpackPartitioner()],
)

exec_prog = edge.to_executorch()

with open("xnnpack_simpleconv.pte", "wb") as file:
    exec_prog.write_to_file(file)


process_image(image_tensor_sample, aten_dialect.module())
