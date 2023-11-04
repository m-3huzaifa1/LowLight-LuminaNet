import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import pytorch_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F

def psnr(target, prediction):
    target = torch.clamp(target, 0.0, 1.0)
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mae = abs(torch.mean(torch.abs(target - prediction)))
    mae_ = mae.item()
    print("MAE :",mae_)
    mse = F.mse_loss(target, prediction)
    mse_ = mse.item()
    print("MSE :",mse_)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value

#def ssim(target, prediction):
    # Assuming you have two PyTorch tensors, input_tensor and output_tensor
# input_tensor represents the ground truth image
# output_tensor represents the generated or enhanced image
#    ssim_value = 1-pytorch_ssim.ssim(target, prediction).item()
#    print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
#    ssim_loss = pytorch_ssim.SSIM()
#    ssim_out = 1-ssim_loss(target, prediction)
#    ssim_value = ssim_out.item()
#    ssim_score = ssim_loss(target, prediction)
#    return ssim_score

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)
  
 
	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	Lumina_Net = model.enhance_net_nopool(scale_factor).cuda()
	Lumina_Net.load_state_dict(torch.load('snapshots_LN/Epoch126.pth'))
	start = time.time()
	enhanced_image,params_maps = Lumina_Net(data_lowlight)

	end_time = (time.time() - start)

	#print(end_time)


	image_path = image_path.replace('test_data','result_LN')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
  
	return end_time , enhanced_image , data_lowlight

if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'data/test_data/'    
        file_list = os.listdir(filePath)
        sum_time = 0
        sum_psnr = 0
        avg_psnr = 0
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*") 
            for image in test_list:
                print(image)
                end_time , enhanced_image , data_lowlight = lowlight(image)
                psnrr = psnr(data_lowlight, enhanced_image)
                print(f"PSNR: {psnrr:.2f}")
                #ssimm = ssim(data_lowlight, enhanced_image)
                #print(f"SSIM: {ssimm:.4f}")
                sum_psnr = sum_psnr + psnrr
                sum_time = sum_time + end_time
            avg_psnr = sum_psnr / len(test_list)
            print("AVG_psnr : ",avg_psnr.item())
        print(sum_time)

