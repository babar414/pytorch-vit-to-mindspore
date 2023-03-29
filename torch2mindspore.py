# convert pytorch model parameters to mindspore model parameters

import numpy as np
import torch
import argparse
import mindspore
from mindspore import Parameter
from mindspore import Tensor
from mindspore import save_checkpoint

def main(args):
    # load torch checkpoint
    pth_ptorch_path = args.model
    torch_pth = torch.load(pth_ptorch_path, map_location='cpu')
    pt_model_params = torch_pth['model']

    params_ms_dict = {} # we start out by empty dictionary and put parameters one by one

    num_blocks = 12
    ms_to_pt_table = {'encoder.patch_embed.projection.weight': 'patch_embed.proj.weight',
                    'encoder.encoder_pos_embedding':'pos_embed',
                    'encoder.cls_token':'cls_token',
                    'encoder.fc_norm.gamma':'fc_norm.weight',
                    'encoder.fc_norm.beta':'fc_norm.bias',
                    'head.weight':'head.weight',
                    'head.bias':'head.bias',
    #                    'encoder.norm.gamma':'norm.weight',
    #                    'encoder.norm.beta':'norm.bias',
                    'encoder.patch_embed.projection.bias' : 'patch_embed.proj.bias'}

    # first copy all the non-encoder parameters (iterate over dictionary)
    for ms_par_name, pt_par_name in ms_to_pt_table.items():
        par_value = pt_model_params[pt_par_name].numpy()
        params_ms_dict[ms_par_name] = Parameter(Tensor(par_value, dtype=mindspore.float32))

    # iterate over all the blocks
    for blk in range(num_blocks):
        
        # copy the layerNorm
        pt_norm1_weight = pt_model_params[f'blocks.{blk}.norm1.weight'].numpy()
        pt_norm2_weight = pt_model_params[f'blocks.{blk}.norm2.weight'].numpy()
        pt_norm1_bias = pt_model_params[f'blocks.{blk}.norm1.bias'].numpy()
        pt_norm2_bias = pt_model_params[f'blocks.{blk}.norm2.bias'].numpy()
        
        norm1_gamma = Tensor(pt_norm1_weight, dtype=mindspore.float32)
        norm2_gamma = Tensor(pt_norm2_weight, dtype=mindspore.float32)
        norm1_beta = Tensor(pt_norm1_bias, dtype=mindspore.float32)
        norm2_beta = Tensor(pt_norm2_bias, dtype=mindspore.float32)
        
        params_ms_dict[f'encoder.encoder.blocks.{blk}.layernorm1.gamma'] = Parameter(norm1_gamma)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.layernorm2.gamma'] = Parameter(norm2_gamma)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.layernorm1.beta'] = Parameter(norm1_beta)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.layernorm2.beta'] = Parameter(norm2_beta)
        
        # copy the attention
        pt_qkv_weight_pt = pt_model_params[f'blocks.{blk}.attn.qkv.weight'].numpy()
        pt_qkv_bias_pt = pt_model_params[f'blocks.{blk}.attn.qkv.bias'].numpy()
        
        pt_qkv_weight = np.split(pt_qkv_weight_pt,3)
        pt_qkv_bias = np.split(pt_qkv_bias_pt,3)


        att_dens1_weight = Tensor(pt_qkv_weight[0], dtype=mindspore.float32)
        att_dens2_weight = Tensor(pt_qkv_weight[1], dtype=mindspore.float32)
        att_dens3_weight = Tensor(pt_qkv_weight[2], dtype=mindspore.float32)
        att_dens1_bias = Tensor(pt_qkv_bias[0], dtype=mindspore.float32)
        att_dens2_bias = Tensor(pt_qkv_bias[1], dtype=mindspore.float32)
        att_dens3_bias = Tensor(pt_qkv_bias[2], dtype=mindspore.float32)

        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense1.weight'] = Parameter(att_dens1_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense2.weight'] = Parameter(att_dens2_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense3.weight'] = Parameter(att_dens3_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense1.bias'] = Parameter(att_dens1_bias)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense2.bias'] = Parameter(att_dens2_bias)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.dense3.bias'] = Parameter(att_dens3_bias)
        
        att_proj_weight = pt_model_params[f'blocks.{blk}.attn.proj.weight'].numpy()
        att_proj_bias = pt_model_params[f'blocks.{blk}.attn.proj.bias'].numpy()
        
        att_proj_weight = np.transpose(att_proj_weight, axes=(1, 0)) # or set transposeB = True for self.proj in MSpore
        att_proj_weight = Tensor(att_proj_weight, dtype=mindspore.float32)
        att_proj_bias = Tensor(att_proj_bias, dtype=mindspore.float32)
        
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.projection.weight'] = Parameter(att_proj_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.attention.projection.bias'] = Parameter(att_proj_bias)

        # copy the MLP
        mlp_fc_1_weight = pt_model_params[f'blocks.{blk}.mlp.fc1.weight'].numpy()
        mlp_fc_2_weight = pt_model_params[f'blocks.{blk}.mlp.fc2.weight'].numpy()
        mlp_fc_1_bias = pt_model_params[f'blocks.{blk}.mlp.fc1.bias'].numpy()
        mlp_fc_2_bias = pt_model_params[f'blocks.{blk}.mlp.fc2.bias'].numpy()

        mlp_fc_1_weight = np.transpose(mlp_fc_1_weight, axes=(1, 0)) # if transposeB = False in MSpore FForward
        mlp_fc_2_weight = np.transpose(mlp_fc_2_weight, axes=(1, 0)) # if transposeB = False in MSpore FForward

        out_map_weight = Tensor(mlp_fc_1_weight, dtype=mindspore.float32)
        out_proj_weight = Tensor(mlp_fc_2_weight, dtype=mindspore.float32)
        out_map_bias = Tensor(mlp_fc_1_bias, dtype=mindspore.float32)
        out_proj_bias = Tensor(mlp_fc_2_bias, dtype=mindspore.float32)
        
        params_ms_dict[f'encoder.encoder.blocks.{blk}.output.mapping.weight'] = Parameter(out_map_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.output.projection.weight'] = Parameter(out_proj_weight)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.output.mapping.bias'] = Parameter(out_map_bias)
        params_ms_dict[f'encoder.encoder.blocks.{blk}.output.projection.bias'] = Parameter(out_proj_bias)


    params_list = []
    for param in params_ms_dict:
        params_list.append({'name':param,
                    'data':params_ms_dict[param].value()})


    # Save the parameters to a checkpoint file
    pth_ptorch_path = pth_ptorch_path.strip().split('/')[-1] if '/' in pth_ptorch_path else pth_ptorch_path
    file_name = f'pt_to_mindspore_conv_{pth_ptorch_path[:-4]}.ckpt'

    print("check_point_saved")
    print(".....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mae_finetuned_vit_base.pth",
                        help="path to PyTorch pth model file")
    
    args = parser.parse_args()
    
    main(args)
