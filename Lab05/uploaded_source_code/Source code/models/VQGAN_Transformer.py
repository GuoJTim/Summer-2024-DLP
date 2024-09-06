import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F
import random


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, q_loss = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0],-1)
        return quant_z, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            #raise Exception('TODO2 step1-2!')
            return lambda r : 1 - r
        elif mode == "cosine":
            #raise Exception('TODO2 step1-2!')
            return lambda r : np.cos(r * np.pi / 2)
        elif mode == "square":
            #raise Exception('TODO2 step1-2!')
            return lambda r : 1 - r ** 2
        else:
            raise NotImplementedError

    def gen_mask(self, sz ,masked_cnt):
        mask = np.zeros(sz, dtype=np.int32)
        
        tables = []
        for i in range(sz[0]):
            for j in range(sz[1]):
                tables.append([i,j])
        random.shuffle(tables)
        for i in range(masked_cnt):
            mask[tables[i][0],tables[i][1]] = 1
        return mask

    

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)

        z_sz = (16,16)
        mask_ratio = self.gamma(np.random.uniform())

        mask = self.gen_mask(z_sz,math.floor((z_sz[0] * z_sz[1])*mask_ratio))
        mask = torch.tensor(mask, dtype=torch.bool).view(-1).unsqueeze(0).to(z_indices.device)
        #print("shape:",z_indices.shape)
        #print("mask:",mask.shape) # 16*16* batch_size
        # z_indices ground truth
        # logits transformer predict the probability of token
        
        codebooks = self.mask_token_id * torch.ones_like(z_indices) # code books

        a_indices = (~mask) * z_indices + mask * codebooks

        logits = self.transformer(a_indices)

        #raise Exception('TODO2 step1-3!')
        # print(logits.shape, z_indices.shape)
        return logits, z_indices, mask
    

    
##TODO3 step1-1: define one iteration decoding   
    ## masked_token Y_M^(t) current masked token from mask_bc
    @torch.no_grad()
    def inpainting(self,z_indices,org_mask_token,masked_token,current_iteration,total_iteration,gamma_func): # iterative decoding
        # current masked_token (mask_bc)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = (~masked_token) * z_indices + masked_token * masked_indices 

        #raise Exception('TODO3 step1-1!')
        logits = self.transformer(a_indices) # get current latent predict 
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        # print("logits shape:",logits.shape)
        logits = F.softmax(logits, dim=-1) # convert to distribution

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits,dim=-1) # get maximum loc as pred

        # 
        ratio= current_iteration / total_iteration
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.empty_like(z_indices_predict_prob).uniform_(0, 1)
        g = -torch.log(-torch.log(g))
        #g = 1
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        masked_confidence = confidence.masked_fill((~masked_token),float('inf')) # set ~mask as inf
        #sort the confidence for the rank 
        sorted_confidence, sorted_indices = torch.sort(masked_confidence, dim=-1) # sort them
        #define how much the iteration remain predicted tokens by mask scheduling

        mask_cnt = int(torch.sum(org_mask_token)) # get masked count


        gamma = self.gamma_func(gamma_func)
        total_mask = int(gamma(ratio)*mask_cnt) # 根據 ratio 逐漸增加
        new_mask = torch.zeros_like(org_mask_token)
        new_mask.scatter_(1, sorted_indices[:, :total_mask], 1) # new mask
        
        
        # print(int(torch.sum(masked_token)),current_iteration,ratio,int(gamma(ratio)),total_mask)
        
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        original_token = z_indices.clone()
        next_token = torch.where(new_mask, z_indices_predict, original_token)
        #print(torch.count_nonzero(masked_token),torch.count_nonzero(new_mask),mask_cnt)
        
        
        return next_token, new_mask,z_indices
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
