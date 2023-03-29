# convert mindspore model parameters to torch model parameters

import numpy as np
import torch
import argparse
import mindspore
from mindspore import Parameter
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore.train.serialization import load_checkpoint
from collections import OrderedDict

def main(args):
    # load mindspore checkpoint
    ckpt_mindspore_path = args.model
    mindspore_ckpt = load_checkpoint(ckpt_mindspore_path)

    params_pt_dict = {} # we start out by empty dictionary and put parameters one by one

    num_blocks = 12

    pt_to_ms_table = {'patch_embed.proj.weight' : 'encoder.patch_embed.projection.weight',
                    'pos_embed' : 'encoder.encoder_pos_embedding',
                    'cls_token' : 'encoder.cls_token',
                    'fc_norm.weight' : 'encoder.fc_norm.gamma',
                    'fc_norm.bias' : 'encoder.fc_norm.beta',
                    'head.weight' : 'head.weight',
                    'head.bias' : 'head.bias',
    #                    'encoder.norm.gamma':'norm.weight',
    #                    'encoder.norm.beta':'norm.bias',
                    'patch_embed.proj.bias':'encoder.patch_embed.projection.bias'}

    # first copy all the non-encoder parameters (iterate over dictionary)
    for pt_par_name, ms_par_name in pt_to_ms_table.items():
        par_value = mindspore_ckpt[ms_par_name].value().asnumpy()
        params_pt_dict[pt_par_name] = torch.Tensor(par_value)
    
    # for pa in params_pt_dict.items():
    #     print(pa)

    # iterate over all the blocks
    for blk in range(num_blocks):
        
        # copy the layerNorm
        norm1_gamma = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.layernorm1.gamma']
        norm2_gamma = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.layernorm2.gamma']
        norm1_beta = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.layernorm1.beta']
        norm2_beta = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.layernorm2.beta']
        
        params_pt_dict[f'blocks.{blk}.norm1.weight'] = torch.tensor(norm1_gamma.asnumpy(), dtype=torch.float32)
        params_pt_dict[f'blocks.{blk}.norm2.weight'] = torch.tensor(norm2_gamma.asnumpy(), dtype=torch.float32)
        params_pt_dict[f'blocks.{blk}.norm1.bias'] = torch.tensor(norm1_beta.asnumpy(), dtype=torch.float32)
        params_pt_dict[f'blocks.{blk}.norm2.bias'] = torch.tensor(norm2_beta.asnumpy(), dtype=torch.float32)
        

        # copy the attention
        att_dens1_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense1.weight']
        att_dens2_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense2.weight']
        att_dens3_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense3.weight']
        att_dens1_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense1.bias']
        att_dens2_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense2.bias']
        att_dens3_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.dense3.bias']

        pt_qkv_weight_pt = np.concatenate((att_dens1_weight.asnumpy(), 
                                            att_dens2_weight.asnumpy(), 
                                            att_dens3_weight.asnumpy()),axis=0)
        
        pt_qkv_bias = np.concatenate((att_dens1_bias.asnumpy(), 
                                            att_dens2_bias.asnumpy(), 
                                            att_dens3_bias.asnumpy()),axis=0)


        params_pt_dict[f'blocks.{blk}.attn.qkv.weight'] = torch.Tensor(pt_qkv_weight_pt)
        params_pt_dict[f'blocks.{blk}.attn.qkv.bias'] = torch.Tensor(pt_qkv_bias)

        att_proj_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.projection.weight']
        att_proj_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.attention.projection.bias']
        
        att_proj_weight = np.transpose(att_proj_weight.asnumpy(), axes=(1, 0)) # or set transposeB = True for self.proj in MSpore
        
        params_pt_dict[f'blocks.{blk}.attn.proj.weight'] = torch.Tensor(att_proj_weight)
        params_pt_dict[f'blocks.{blk}.attn.proj.bias'] = torch.Tensor(att_proj_bias.asnumpy())

        # copy the MLP
        
        out_map_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.output.mapping.weight']
        out_proj_weight = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.output.projection.weight']
        out_map_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.output.mapping.bias']
        out_proj_bias = mindspore_ckpt[f'encoder.encoder.blocks.{blk}.output.projection.bias']

        out_map_weight = np.transpose(out_map_weight.asnumpy(), axes=(1, 0)) # if transposeB = False in MSpore FForward
        out_proj_weight = np.transpose(out_proj_weight.asnumpy(), axes=(1, 0)) # if transposeB = False in MSpore FForward


        params_pt_dict[f'blocks.{blk}.mlp.fc1.weight'] = torch.Tensor(out_map_weight)
        params_pt_dict[f'blocks.{blk}.mlp.fc2.weight'] = torch.Tensor(out_proj_weight)
        params_pt_dict[f'blocks.{blk}.mlp.fc1.bias'] = torch.Tensor(out_map_bias.asnumpy())
        params_pt_dict[f'blocks.{blk}.mlp.fc2.bias'] = torch.Tensor(out_proj_bias.asnumpy())


    # Save the parameters to a checkpoint file
    state_dic = {key : value for key , value in params_pt_dict.items()}

    state_dic = OrderedDict([('model',state_dic)])
    pth_ptorch_path = ckpt_mindspore_path.strip().split('/')[-1] if '/' in ckpt_mindspore_path else ckpt_mindspore_path
    file_name = f'mindspore_to_pt_conv_{pth_ptorch_path[:-5]}.pth'
    torch.save(state_dic, file_name)

    print("check_point_saved")
    print(".....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mindspore_vit_model.ckpt",
                        help="path to PyTorch pth model file")
    
    args = parser.parse_args()
    
    main(args)
