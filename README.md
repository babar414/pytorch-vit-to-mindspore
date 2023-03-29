# pytorch-vit-to-mindspore
This repository contains python script to convert pytorch-vit [(timm)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) model weights into mindspore vit model weights and vice versa. The conversion is verified by converting MAE Imagnet-1k fine-tuned ViT model weights from PyTorch to Mindspore and then evaluating and comparing the top-1 accuracy. 


## Dependencies

The following dependencies are required to run this script:

- torch
- MindSpore
- numpy
- argparse

## Usage

To convert a PyTorch ViT model to a MindSpore ViT model, run the following command:
```
python torch2mindspore.py --model=<path/to/pytorch/vit/model>
```
Replace `<path/to/pytorch/vit/model>` with the path to your PyTorch ViT model.

The converted model will be saved with the filename prefixed with words "pt_to_mindspore_conv_" in the same directory as the PyTorch model.


Similarly, to convert a Mindspore ViT model to a PyTorch ViT model, run the following command:
```
python mindspore2torch.py --model=<path/to/mindspore/vit/model>
```
Replace `<path/to/mindspore/vit/model>` with the path to your Mindspore ViT model.

The converted model will be saved with the filename prefixed with words "mindspore_to_pt_conv_" in the same directory as the PyTorch model.

## Example
```
python torch2mindspore.py --model=mae_finetuned_vit_base.pth
```
or
```
python mindspore2torch.py --model=mae_finetuned_vit_base.ckpt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
