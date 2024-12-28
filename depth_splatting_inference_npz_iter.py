'''
huggingface-cli download svjack/Genshin-Impact-Novel-Video Genshin-Impact-Cutness-video1.zip --repo-type dataset --local-dir .

##### choose this
python depth_splatting_inference_npz_iter.py \
   --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1\
   --unet_path ./weights/DepthCrafter --process_length 64 \
   --input_path test_videos0 \
   --output_path test_videos0_splatting_64

python npz_to_video.py --input_folder test_videos0_splatting_64 --output_folder test_videos0_splatting_64_video
'''

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from diffusers.training_utils import set_seed
from decord import VideoReader, cpu

from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth, read_video_frames

from Forward_Warp import forward_warp

save_grid = True

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if cpu_offload is not None:
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = True,
    ):
        set_seed(seed)

        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]

        res = res.sum(-1) / res.shape[-1]
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        res = F.interpolate(tensor_res, size=(original_height, original_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:,0,:,:]
        res = (res - res.min()) / (res.max() - res.min())
        vis = vis_sequence_depth(res)

        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            np.savez_compressed(save_path + ".npz", depth=res)            
            np.savez_compressed(save_path + "_depth_vis.npz", video=vis)

        return res, vis
    

class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map
        

def process_video(
    input_video_path: str,
    output_video_path: str,
    depthcrafter_demo: DepthCrafterDemo,
    max_disp: float = 20.0,
    process_length: int = -1
):
    video_depth, depth_vis = depthcrafter_demo.infer(
        input_video_path,
        output_video_path,
        process_length,
    )

    if save_grid:
        vid_reader = VideoReader(input_video_path, ctx=cpu(0))
        original_fps = vid_reader.get_avg_fps()
        input_frames = vid_reader[:process_length].asnumpy() / 255.0
    
        if process_length != -1 and process_length < len(input_frames):
            input_frames = input_frames[:process_length]
    
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
    
        left_video = torch.tensor(input_frames).permute(0, 3, 1, 2).float().contiguous().cuda()
        disp_map   = torch.tensor(video_depth).unsqueeze(1).float().contiguous().cuda()
    
        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp
    
        right_video, occlusion_mask = stereo_projector(left_video, disp_map)
    
        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)
    
        video_grid_top = np.concatenate([input_frames, depth_vis], axis=2)
        video_grid_bottom = np.concatenate([occlusion_mask, right_video], axis=2)
        video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=1)
        
        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )
        np.savez_compressed(save_path + "_video_grid.npz", video_grid=video_grid)
    
        print(f"Finished processing {input_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Process videos using DepthCrafter.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or directory containing videos.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to the UNet model.")
    parser.add_argument("--pre_trained_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--max_disp", type=float, default=20.0, help="Maximum disparity value.")
    parser.add_argument("--process_length", type=int, default=-1, help="Number of frames to process.")
    args = parser.parse_args()

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=args.unet_path,
        pre_trained_path=args.pre_trained_path,
        cpu_offload = "sequential"
    )

    if os.path.isfile(args.input_path):
        video_files = [args.input_path]
    else:
        video_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    os.makedirs(args.output_path, exist_ok=True)

    for video_file in tqdm(video_files, desc="Processing videos"):
        output_video_name = os.path.basename(video_file).replace(" ", "_")
        output_video_path = os.path.join(args.output_path, output_video_name)
        process_video(video_file, output_video_path, depthcrafter_demo, args.max_disp, args.process_length)


if __name__ == "__main__":
    main()
