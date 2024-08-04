import torch
import random

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from ldm.util import instantiate_from_config
from itertools import combinations


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sampling(model, sampler, prompt, n_samples, scale=7.5, steps=50, conjunction=False, mask_cond=None, img=None):
    H = W = 512
    C = 4
    f = 8
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in range(n_samples):
                    for bid, p in enumerate(prompt):
                        
                        uc = model.get_learned_conditioning([""])
                        _c = model.get_learned_conditioning(p)
                        c = {'k': [_c], 'v': [_c]}
                        shape = [C, H // f, W // f]
                        
                        samples_ddim, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=0.0,
                                                            x_T=img,
                                                            quiet=True,
                                                            mask_cond = mask_cond,
                                                            save_attn_maps=True)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        all_samples.append(x_checked_image_torch)
    return all_samples


def diff(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference

def intersection(t1, t2):
    i = np.intersect1d(t1, t2)
    return torch.from_numpy(i) 

def priority_for_1_class(pixel_avai, cls_mask):
    assert len(pixel_avai) == 1
    rank_1 = cls_mask[0][pixel_avai[0].sort(descending=True)[1]]
    return [rank_1]

def priority_for_2_class(pixel_avai, cls_mask):
    assert len(pixel_avai) == 2
    rank_1 = cls_mask[0][pixel_avai[0].sort(descending=True)[1]]
    rank_2 = cls_mask[1][pixel_avai[1].sort(descending=True)[1]]
    
    priority_for_1 = torch.cat((rank_1, rank_2.flip(dims=(0,))),0)
    priority_for_2 = torch.cat((rank_2, rank_1.flip(dims=(0,))),0)
    return [priority_for_1, priority_for_2]

def block(value, scale_factor=4):
    vs = []
    for v in value:
        e = torch.zeros(256)
        e[v] = 1
        e = rearrange(e, '(w h)-> w h', w=16)
        e_resized = F.interpolate(e.reshape(1,1,16,16), scale_factor=scale_factor)[0][0]
        e_resized = rearrange(e_resized, 'w h -> (w h)')
        vs.append(torch.where(e_resized==1)[0])
    return vs

    
def block_single_pixel(value, scale_factor=4):
    e = torch.zeros(256)
    e[value] = 1
    e = rearrange(e, '(w h)-> w h', w=16)
    e_resized = F.interpolate(e.reshape(1,1,16,16), scale_factor=scale_factor)[0][0]
    e_resized = rearrange(e_resized, 'w h -> (w h)')
    return torch.where(e_resized==1)[0]

def generate(img_, prompt, model, sampler, mask_cond={'is_use': False}):
    ddim_steps = 50
    n_samples = 1
    scale = 7.5
    all_samples = sampling(model, sampler, prompt, 
                            n_samples, scale, 
                            ddim_steps, mask_cond=mask_cond, conjunction=False, img=img_)
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=int(np.sqrt(n_samples)))
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    attn_maps = [item[0][0] for item in sampler.attn_maps['input_blocks.8.1.transformer_blocks.0.attn2']]
    maps = [torch.mean(item, axis=0) for item in attn_maps]
    maps = [rearrange(item, 'w h d -> d w h')[None,:] for item in maps]
    maps = rearrange(torch.cat(maps,dim=0), 't word w h -> word t w h')
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(maps[2][0])
    plt.axis("off")
    plt.subplot(1, 5, 3)
    plt.imshow(maps[5][0])
    plt.axis("off")
    plt.subplot(1, 5, 4)
    plt.imshow(maps[2][-1])
    plt.axis("off")
    plt.subplot(1, 5, 5)
    plt.imshow(maps[5][-1])
    plt.axis("off")
    plt.show()
    

class initializer:
    def __init__(self, model, sampler, normalize=True):
        self.model = model
        self.sampler =sampler
        self.H = 512
        self.W = 512
        self.C = 4
        self.f = 8
        self.shape = [1, self.C, self.H // self.f, self.W // self.f]
        self.cond = {'is_use': False}
        
    
    def get_attn(self, prompt, img, scale=7.5, steps=50):
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for bid, p in enumerate(prompt):
                        uc = self.model.get_learned_conditioning([""])
                        kv = self.model.get_learned_conditioning(p)
                        c = {'k':[kv], 'v': [kv]}
                        shape = [self.C, self.H // self.f, self.W // self.f]
                        self.sampler.get_attention(S=steps,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=img,
                                        quiet=True,
                                        mask_cond=self.cond,
                                        save_attn_maps=True)
        all_attn_maps = [item[0][0] for item in self.sampler.attn_maps['input_blocks.8.1.transformer_blocks.0.attn2']]
        avg_maps = [torch.mean(item, axis=0) for item in all_attn_maps]
        avg_maps = [rearrange(item, 'w h d -> d w h')[None,:] for item in avg_maps]
        avg_maps = rearrange(torch.cat(avg_maps,dim=0), 't word w h -> word t w h')
        return avg_maps
    
    def init_image(self, prompt, indicies):
        img = torch.randn(self.shape).cuda()
        maps = [self.get_attn(prompt, img)[i][0] for i in indicies]
        return img, maps
    
    def preprocess_map(self, maps):
        maps = [m / m.mean() for m in maps]
        return maps

    def pixel_classify(self, maps):
        maps = torch.stack(maps, dim=0)
        pixel_class = torch.argmax(maps, dim=0)
        return pixel_class
    
    def generate_region_mask(self, regions):
        region_mask = []
        
        for i in range(len(regions)):
            z = torch.zeros(16,16)
            r = regions[i]
            z[r[1]:r[3], r[0]:r[2]]=1
            region_mask.append(z)
        region_mask = rearrange(torch.stack(region_mask, dim=0), 'n w h -> n (w h)')
        if len(regions)==2:
            mask_1 = torch.where(region_mask[0] == 1)[0]
            mask_2 = torch.where(region_mask[1] == 1)[0]
            if mask_1.shape[0] > mask_2.shape[0]:
                region_mask[0][intersection(mask_1, mask_2)] = 0
            else:
                region_mask[1][intersection(mask_1, mask_2)] = 0
        return region_mask
        
    def assign_to_region(self, maps, pixel_class, region_mask):
        space = range(len(region_mask))
        
        pixel_class = rearrange(pixel_class,  'w h -> (w h)')
        maps = [rearrange(m, 'w h -> (w h)') for m in maps]
        cls_mask = [torch.where(pixel_class == i)[0] for i in space]
        
        pixel_avai = [maps[i][cls_mask[i]] for i in space]
        num_avai = [mask.shape[0] for mask in cls_mask]
        num_need = [torch.where(r == 1)[0].shape[0] for r in region_mask]
        if len(region_mask) == 1:
            pri = priority_for_1_class(pixel_avai, cls_mask)
        if len(region_mask) == 2:
            pri = priority_for_2_class(pixel_avai, cls_mask)
        to_obj = [pri[i][:num_need[i]] for i in space]
        return to_obj
    
    def region_to_block(self, mask):
        # [0,0,0,....,1,1,1,1,......]
        # input:
        # [[0,0,0,.....0,0,0],
        #  [0,0,0,.....0,0,0],
        #  [1,1,1,.....0,0,0],
        #  [1,1,1,.....0,0,0],
        #  [1,1,1,.....0,0,0]]
        
        if mask.unique().shape < mask.shape:
            r = []
            index = torch.where(mask)[0]
            for p in index:
                r.append(block_single_pixel(p))
        else:
            r = []
            for p in mask:
                r.append(block_single_pixel(p))
        return r
    
    def create_img(self, img, region_mask, to_obj):
        # input: img, regions of objects, coordinates of pixels expected to be put into regions
        img = img.clone()[0]
        new_img = torch.zeros(img.shape).cuda()
        w = new_img.shape[1]
        img = rearrange(img, 'c w h -> c (w h)')
        new_img = rearrange(new_img, 'c w h -> c (w h)')
        # inside region
        in_region_block = [self.region_to_block(mask) for mask in region_mask]
        # out-side region
        out_mask = torch.where(region_mask.sum(axis=0)==0)[0]
        out_region_block = self.region_to_block(out_mask)
        # in-side pixel
        inside = [self.region_to_block(mask) for mask in to_obj]
        # out-side pixel
        outside = self.region_to_block(diff(torch.arange(256), torch.concat(to_obj,0)))

        for i in range(len(region_mask)):
            for j in range(len(in_region_block[i])):
                for k in range(in_region_block[i][j].shape[0]):
                    new_img[:, in_region_block[i][j][k]] = img[:, inside[i][j][k]]
        for i in range(len(outside)):
            for k in range(outside[i].shape[0]):
                new_img[:, out_region_block[i][k]] = img[:, outside[i][k]]
        new_img = rearrange(new_img, 'c (w h) -> c w h', w = w)
        return new_img[None, :]
    
    def make_image(self, prompt, regions, indicies, img=None, visualize=False):
        if img == None:
            img, maps = self.init_image(prompt, indicies)
        else:
            maps_all = self.get_attn(prompt, img)
            maps = [maps_all[i][0] for i in indicies]
        region_mask = self.generate_region_mask(regions)
        maps = self.preprocess_map(maps)
        pixel_class = self.pixel_classify(maps)
        to_obj = self.assign_to_region(maps, pixel_class, region_mask)
        img_ = self.create_img(img, region_mask, to_obj)
        if visualize:
            # original attention maps
            map_1 = F.interpolate(maps[0].reshape(1,1,16,16), scale_factor=4)[0][0]
            map_2 = F.interpolate(maps[1].reshape(1,1,16,16), scale_factor=4)[0][0]
            plt.subplot(1, 2, 1)
            plt.imshow(map_1)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(map_2)
            plt.axis("off")
            plt.show()
            # modified attention maps
            rescaled_maps = torch.stack([map_1, map_2],0).cuda()[None, :]
            rescaled_maps = self.create_img(rescaled_maps, region_mask, to_obj).cpu()[0]
            plt.subplot(1, 2, 1)
            plt.imshow(rescaled_maps[0])
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(rescaled_maps[1])
            plt.axis("off")
            plt.show()
        return img_  
        
    def exchange_map_perfect_match(self, img, to_obj, region_mask, bkg_ind):
        bkg_block = block_single_pixel(bkg_ind)
        area = [torch.where(mask)[0] for mask in region_mask]
        go_to_obj = []
        get_out_obj = []
        for i in range(len(to_obj)):
            already_in = intersection(to_obj[i], area[i])
            go_to_obj.append(diff(to_obj[i], already_in))
            get_out_obj.append(diff(area[i], already_in))
        # to block
        should_in = [block(pixels) for pixels in go_to_obj]
        should_out = [block(pixels) for pixels in get_out_obj]
        img_flatten = rearrange(img.clone(), 'n c w h -> n c (w h)')
        img_flatten_source = rearrange(img.clone(), 'n c w h -> n c (w h)')
        for i in range(len(should_in)):
            in_blocks = should_in[i]
            out_blocks = should_out[i]
            for j in range(len(in_blocks)):
                img_flatten[:, :, out_blocks[j]]=img_flatten_source[:, :, in_blocks[j]]
                if len(should_in) == 1:
                    img_flatten[:, :, in_blocks[j]]=img_flatten_source[:, :, out_blocks[j]]
        if len(should_in) == 2:
            erase = []
            erase.append(diff(go_to_obj[0], intersection(go_to_obj[0], area[1])))
            erase.append(diff(go_to_obj[1], intersection(go_to_obj[1], area[0])))
            erase = torch.cat([erase[0], erase[1]], 0)

            get_out_obj[0] = diff(get_out_obj[0], intersection(get_out_obj[0], go_to_obj[1]))
            get_out_obj[1] = diff(get_out_obj[1], intersection(get_out_obj[1], go_to_obj[0]))
            bad_pixels = torch.cat([get_out_obj[0], get_out_obj[1]], 0)

            should_erase = block(erase)
            bad_pixels = block(bad_pixels)
            for i in range(len(should_erase)):
                img_flatten[:, :, should_erase[i]] = img_flatten_source[:, :, bad_pixels[i]]
        img_new = rearrange(img_flatten.clone(), 'n c (w h) -> n c w h', w=64)
        return img_new
    
    
    def exchange_perfect(self, prompt, regions, indicies, img=None, visualize=False):
        if img == None:
            img, maps = self.init_image(prompt, indicies)
        else:
            maps_all = self.get_attn(prompt, img)
            maps = [maps_all[i][0] for i in indicies]
            
        # Low-score block
        bkg_block = rearrange(torch.stack(maps,0).sum(0), 'w h -> (w h)').sort()[1][0]
        
        region_mask = self.generate_region_mask(regions)
        maps = self.preprocess_map(maps)
        pixel_class = self.pixel_classify(maps)
        to_obj = self.assign_to_region(maps, pixel_class, region_mask)
        img_new = self.exchange_map_perfect_match(img, to_obj, region_mask, bkg_block)
        if visualize:
            map_1 = F.interpolate(maps[0].reshape(1,1,16,16), scale_factor=4)[0][0]
            map_2 = F.interpolate(maps[1].reshape(1,1,16,16), scale_factor=4)[0][0]

            rescaled_maps = torch.stack([map_1, map_2],0).cuda()[None, :]
            rescaled_maps = self.exchange_map_perfect_match(rescaled_maps, to_obj, region_mask, bkg_block).cpu()[0]
            plt.subplot(1, 2, 1)
            plt.imshow(rescaled_maps[0])
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(rescaled_maps[1])
            plt.axis("off")
            plt.show()
        return img_new
