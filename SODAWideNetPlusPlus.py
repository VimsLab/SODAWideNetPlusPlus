import torch.nn as nn
import torch
import os
import torchvision
import torch.nn.functional as F
import random
from torch.nn import init
import math
from torch.autograd import Variable
from math import exp
import numpy as np


class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, padding = kernel_size // 2, dilation = 1),
			nn.BatchNorm2d(mid_channels),
			nn.GELU(),
			nn.Conv2d(mid_channels, out_channels, kernel_size= kernel_size, padding=kernel_size // 2, dilation=1),
			nn.BatchNorm2d(out_channels),
			nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, mid_channels = None):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
		nn.MaxPool2d(2),
		DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, bilinear=True):
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, kernel_size, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels, kernel_size)


	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
		diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

class DoubleConvMod(nn.Module):
	"""(convolution => [GN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		if mid_channels % 32 != 0:
			groups_n = 16
		else:
			groups_n = 32
		self.double_conv = nn.Sequential(
		nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, stride=stride, padding=padding, dilation=dilation, groups = groups),
		nn.GroupNorm(groups_n, mid_channels),
		nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class DoubleConvMody(nn.Module):
	"""(convolution => [BN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
		nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, stride=stride,padding=padding, dilation=dilation, groups = groups),
		nn.BatchNorm2d(mid_channels),
		nn.GELU(),
		)

	def forward(self, x):
		return self.double_conv(x)

def isPowerOfTwo(n):
	if (n == 0):
		return False
	while (n != 1):
		if (n % 2 != 0):
			return False
		n = n // 2
	return True

class DoubleConvL(nn.Module):
	"""(convolution => [GN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		if out_channels < 32:
			if out_channels < 16:
				groups_n = 8
			else:
				groups_n = 16
		else:
			groups_n = 32
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = 1),
			nn.GroupNorm(groups_n, mid_channels),
			nn.GELU(),
			nn.Conv2d(mid_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = 1),
			nn.GroupNorm(groups_n, out_channels),
			nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class AGLRFE(nn.Module):
	def __init__(self, ns, in_channels, out_channels, mid_channels = 32, factor = 4, dilation_rates = None):
		super(AGLRFE, self).__init__()
		print("Dilation Rates - ", dilation_rates, flush = True)
		self.layers_final = nn.ModuleList([])
		self.up = nn.ModuleList([])
		self.ns = ns
		self.out_channels = out_channels
		self.rescon = DoubleConv(in_channels, mid_channels)
		self.attn_pre = nn.Sequential(
			nn.AvgPool2d(factor, factor),
			DoubleConv(in_channels, mid_channels)
		)

		self.attn = Self_Attn_MRFFAM(mid_channels)
		if self.ns >= 1:
			for i in dilation_rates:
				self.layers_final.append(DoubleConvL(mid_channels, mid_channels, 3, 1, i, i))
				self.up.append(
					nn.Sequential(
						nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
						DoubleConv(mid_channels, 1)
					)
				)
			self.conv = Down((ns + 1) * mid_channels, self.out_channels)

	def forward(self, x):
		ns = self.ns
		attn_weight_c = self.attn_pre(x)
		attn_weight = self.attn(attn_weight_c)
		x = self.rescon(x)
		res = torch.cat([k(x) * (1. + torch.sigmoid(weights(attn_weight))) for i, k, weights in zip(range(len(self.layers_final)), self.layers_final, self.up)], dim = 1)
		return self.conv(torch.cat([res, x], dim = 1))

class MRFFAM(nn.Module):

	def __init__(self, ns, in_channels, out_channels, mid_channels = 32, dilation_rates = None):
		super(MRFFAM, self).__init__()
		self.layers_final = nn.ModuleList([])
		self.ns = ns
		self.in_channels = in_channels // 2
		self.rescon = DoubleConv(self.in_channels, mid_channels)
		if self.ns >= 1:
			for i in dilation_rates:
				self.layers_final.append(DoubleConvL(mid_channels, mid_channels, 3, 1, i, i))
			self.conv = DoubleConv((ns + 1) * mid_channels, self.in_channels)

	def forward(self, x):
		ns = self.ns
		x = self.rescon(x)
		res = torch.cat([k(x) for i, k in zip(range(len(self.layers_final)), self.layers_final)], dim = 1)
		return self.conv(torch.cat([res, x], dim = 1))

class CombinationCFM(nn.Module):
	def __init__(self, channels, mid = 64):
		super(CombinationCFM, self).__init__()
		self.localcomb = DoubleConvMod(channels, mid)
		self.globalcomb = DoubleConvMod(channels, mid)
		self.localmul = DoubleConvMod(mid, mid)
		self.globalmul = DoubleConvMod(mid, mid)
		self.localaft = DoubleConvMod(mid, mid)
		self.globalaft = DoubleConvMod(mid, mid)
		self.final = DoubleConvMod(2 * mid, channels)

	def forward(self, l, g):

		localcomb, globalcomb = self.localcomb(l), self.globalcomb(g)
		localmul, globalmul = self.localmul(localcomb), self.globalmul(globalcomb)
		res = localmul + globalmul
		local_aft, global_aft = self.localaft(res), self.globalaft(res)
		new_local = localcomb + local_aft
		new_global = globalcomb + global_aft
		return self.final(torch.cat([new_local, new_global], dim = 1)) + l + g

class ALPM(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ALPM, self).__init__()
		self.local = nn.MaxPool2d(2, 2)
		self.localu = Down(in_channels, in_channels)
		self.attn = Self_Attn_MRFFAM(in_channels)
		self.localup = Up(in_channels * 2, out_channels)
		self.final = DoubleConv(in_channels, out_channels)
	
	def forward(self, x):
		res = self.local(x)
		res_ = self.localu(res)
		res__ = self.localup(res_ + self.attn(res_), res)
		return res__ + self.final(res)

class Self_Attn_MRFFAM(nn.Module):
	""" Self attention Layer"""
	def __init__(self,in_dim,factor=1,padding=1,dilation=1,sr_ratio=1):
		super(Self_Attn_MRFFAM,self).__init__()
		self.factor = factor
		self.chanel_in = in_dim
		if sr_ratio > 1:
			self.key_conv = DoubleConvMod(in_dim ,in_dim//factor , kernel_size= sr_ratio, stride = sr_ratio, padding = 0, dilation=1)
			self.value_conv = DoubleConvMod(in_dim ,in_dim , kernel_size= sr_ratio, stride = sr_ratio, padding = 0, dilation=1)
		else:
			self.key_conv = DoubleConvMod(in_dim ,in_dim//factor , kernel_size= 1, padding = 0, dilation=1)
			self.value_conv = DoubleConvMod(in_dim ,in_dim , kernel_size= 1, padding = 0, dilation=1)

		self.query_conv = DoubleConvMod(in_dim , in_dim//factor , kernel_size= 1, padding = 0, dilation=1)
		# self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1)
		self.C = in_dim
		self.sr_ratio = sr_ratio

	def forward(self, x):
		m_batchsize,C,width ,height = x.size()
		# m_batchsize_k,C_k,width_k ,height_k = y.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width // self.sr_ratio * height//self.sr_ratio) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key)
		attention = self.softmax(energy / ((self.chanel_in // self.factor) ** 0.5)) # BX (N) X (N)
		proj_value = self.value_conv(x).view(m_batchsize,-1,width//self.sr_ratio*height//self.sr_ratio) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,self.C,width,height)

		# out = self.gamma*out
		return out

class SODAWideNetPlusPlus(nn.Module):

	def __init__(self, 
	n_channels, 
	n_classes, 
	bilinear = True, 
	use_contour = False,
	deep_supervision = False, 
	ssl = False, 
	factorw = 1,
	dilation_rates = [[1, 2, 3, 4, 5], [1, 2, 3, 4]]):

		super(SODAWideNetPlusPlus, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear
		self.ssl = ssl
		self.inc = nn.Sequential(
			DoubleConv(n_channels, 32 * factorw),
			Down(32 * factorw, 32 * factorw)
		)

		print("Dilation Rates - ", dilation_rates, flush = True)

		self.down1 = AGLRFE((len(dilation_rates[0])), 32 * factorw, 64 * factorw, 32 * factorw, dilation_rates = dilation_rates[0])
		factor = 2 if bilinear else 1
		self.down2 = AGLRFE((len(dilation_rates[1])), 64 * factorw, (128 * factorw) // factor, 32 * factorw, 2, dilation_rates = dilation_rates[1])

		self.up3 = Up(128 * factorw, (64 * factorw) // factor, bilinear = bilinear)
		self.up4 = Up(64 * factorw, 32 * factorw, bilinear = bilinear)

		self.up3x = MRFFAM((len(dilation_rates[1])), 128 * factorw, (64 * factorw) // factor, 16 * factorw, dilation_rates[1])
		self.up4x = MRFFAM((len(dilation_rates[0])), 64 * factorw, 32 * factorw, 16 * factorw, dilation_rates[0])

		self.outc = OutConv(32 * factorw, n_classes); self.outs = OutConv(32 * factorw, n_classes)

		self.f1 = ALPM(32 * factorw, 64 * factorw)
		self.f2 = ALPM(64 * factorw, (128 * factorw) // factor)

		self.comb1 = CombinationCFM(64 * factorw, 32 * factorw)
		self.comb2 = CombinationCFM((128 * factorw) // factor, 32 * factorw)

		self.combd1 = CombinationCFM((64 * factorw) // factor, 32 * factorw)
		self.combd2 = CombinationCFM((128 * factorw) // factor, 32 * factorw)


		self.contour = use_contour
		self.deep_supervision = deep_supervision

		if self.deep_supervision:
			self.outsdd = OutConv(64 * factorw, n_classes); self.outsrdd = OutConv((128 * factorw) // factor, n_classes)
			self.outl3 = OutConv((128 * factorw) // factor, n_classes)
			self.outl2 = OutConv(64 * factorw, n_classes)

			self.dec2 = OutConv((128 * factorw) // factor, n_classes)
			self.dec1 = OutConv((64 * factorw) // factor, n_classes)

			self.outlpm2 = OutConv((128 * factorw) // factor, n_classes)
			self.outlpm1 = OutConv(64 * factorw, n_classes)

		if self.contour:

			self.outcontour1 = OutConv(32 * factorw, n_classes)
			self.outcontour2 = OutConv(32 * factorw, n_classes)
			
			self.dec2contour = OutConv((128 * factorw) // factor, n_classes)
			self.dec1contour = OutConv((64 * factorw) // factor, n_classes)

		self.up2b = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
		self.up3b = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
		self.up4b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

	def forward(self, x):

		saliency = []
		contours = []

		x1 = self.inc(x)

		res1 = self.f1(x1)
		ds1 = self.down1(x1)
		x2 = self.comb1(res1, ds1)

		res2 = self.f2(x2)
		ds2 = self.down2(x2)
		x3 = self.comb2(res2, ds2)

		if self.ssl:
			ssl = [x1, x2, x3]

		if self.deep_supervision:

			ll_smallest = self.up3b(self.outsdd(ds1))
			saliency.append(ll_smallest)

			ll_small = self.up2b(self.outsrdd(ds2))
			saliency.append(ll_small)

			lpm1_ = self.up3b(self.outlpm1(res1))
			saliency.append(lpm1_)

			lpm2_ = self.up2b(self.outlpm2(res2))
			saliency.append(lpm2_)

			l2_ = self.up3b(self.outl2(x2))
			saliency.append(l2_)

			l3_ = self.up2b(self.outl3(x3))
			saliency.append(l3_)

		if self.contour:

			decds2 = self.up3x(x3)
			x = self.up3(self.combd2(x3, decds2), x2); logits_small = self.up3b(self.outs(x)); contour_small = self.up3b(self.outcontour2(x))

			if self.deep_supervision:
				decds2_o = self.up2b(self.dec2(decds2))
				saliency.append(decds2_o)
			decds2_oc = self.up2b(self.dec2contour(decds2))
			contours.append(decds2_oc)
			
			saliency.append(logits_small)
			contours.append(contour_small)
			

			decds1 = self.up4x(x)
			x = self.up4(self.combd1(x, decds1), x1)
			if self.deep_supervision:
				decds1_o = self.up3b(self.dec1(decds1))
				saliency.append(decds1_o)
			decds1_oc = self.up3b(self.dec1contour(decds1))
			contours.append(decds1_oc)

			logits = self.up4b(self.outc(x)) + logits_small
			contour_s = self.up4b(self.outcontour1(x)) + contour_small
			
			saliency.append(logits)
			contours.append(contour_s)

			if self.ssl:
				return ssl, contours, saliency

			return contours, saliency

		else:

			x = self.up3(self.combd2(x3, self.up3x(x3)), x2); logits_small = self.up3b(self.outs(x))

			saliency.append(logits_small)
			
			x = self.up4(self.combd1(x, self.up4x(x)), x1)
			logits = self.up4b(self.outc(x)) + logits_small

			saliency.append(logits)
			return saliency

