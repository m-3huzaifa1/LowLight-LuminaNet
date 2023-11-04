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
import Loss_functions
import numpy as np
from torchvision import transforms

avg = 0.00

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor
	Lumina_Net = model.enhance_net_nopool(scale_factor).cuda()
	print(sum(p.numel() for p in Lumina_Net.parameters()))

	# Lumina_Net.apply(weights_init)
	#if config.load_pretrain == True:
	#    Lumina_Net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


	L_color = Loss_functions.L_color()
	L_spa = Loss_functions.L_spa()
	L_exp = Loss_functions.L_exp(16)
	# L_exp = Myloss.L_exp(16,0.6)
	L_TV = Loss_functions.L_TV()


	optimizer = torch.optim.Adam(Lumina_Net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	#print(optimizer)
	
	Lumina_Net.train()
	sum_loss = 0.00

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			E = 0.6

			enhanced_image,A  = Lumina_Net(img_lowlight)
			Loss_TV = 1600*L_TV(A)
			# Loss_TV = 200*L_TV(A)			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image,E))

			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp



			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(Lumina_Net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				sum_loss = sum_loss + loss.item()
				print(epoch,":",iteration+1,"=>", loss.item())
			if((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(Lumina_Net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
		avg = sum_loss/25.00
		print("avg loss of ",epoch,':',avg)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=127)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--scale_factor', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots_LN/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots_LN/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	









	
