# pytorch-vit-to-mindspore
This repository contains python script to convert pytorch-vit [(timm)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) model weights into mindspore vit model weights. The conversion is verified by converting MAE Imagnet-1k fine-tuned ViT model weights from PyTorch to Mindspore and then evaluating and comparing the top-1 accuracy. 


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

The converted MindSpore model will be saved with the filename prefixed with words "mindspore_conv_" in the same directory as the PyTorch model.

## Example
```
python torch2mindspore.py --model=mae_finetuned_vit_base.pth
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.
