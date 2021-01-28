import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Affinity_Propagate(nn.Module):

	def __init__(self,
				 prop_time,
				 prop_kernel,
				 norm_type='8sum'):
		"""
		Inputs:
			prop_time: how many steps for CSPN to perform
			prop_kernel: the size of kernel (current only support 3x3)
			way to normalize affinity
				'8sum': normalize using 8 surrounding neighborhood
				'8sum_abs': normalization enforcing affinity to be positive
							This will lead the center affinity to be 0
		"""
		super(Affinity_Propagate, self).__init__()
		self.prop_time = prop_time
		self.prop_kernel = prop_kernel
		assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

		self.norm_type = norm_type
		assert norm_type in ['8sum', '8sum_abs']

		self.in_feature = 1
		self.out_feature = 1


	def forward(self, guidance, blur_mask, sparse_mask=None):
		#print("guidance: ", guidance.shape)
		#print("blur_mask: ", blur_mask.shape)
		self.sum_conv = nn.Conv3d(in_channels=8,
								  out_channels=1,
								  kernel_size=(1, 1, 1),
								  stride=1,
								  padding=0,
								  bias=False)
		weight = torch.ones(1, 8, 1, 1, 1).cuda()
		self.sum_conv.weight = nn.Parameter(weight)
		for param in self.sum_conv.parameters():
			param.requires_grad = False

		gate_wb, gate_sum = self.affinity_normalization(guidance)

		# pad input and convert to 8 channel 3D features
		raw_mask_input = blur_mask

		#blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
		result_mask = blur_mask

		if sparse_mask is not None:
			sparse_mask = sparse_mask.sign()

		for i in range(self.prop_time):
			# one propagation
			spn_kernel = self.prop_kernel
			result_mask = self.pad_blur_mask(result_mask)
			#print("gate_wb: ", gate_wb.shape)
			#print("result_mask: ", result_mask.shape)
			neigbor_weighted_sum = self.sum_conv(gate_wb * result_mask)
			neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
			neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
			result_mask = neigbor_weighted_sum

			if '8sum' in self.norm_type:
				result_mask = (1.0 - gate_sum) * raw_mask_input + result_mask
			else:
				raise ValueError('unknown norm %s' % self.norm_type)

			if sparse_mask is not None:
				result_mask = (1 - sparse_mask) * result_mask + sparse_mask * raw_mask_input

		return result_mask

	def affinity_normalization(self, guidance):

		# normalize features
		if 'abs' in self.norm_type:
			guidance = torch.abs(guidance)

		gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
		gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
		gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
		gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
		gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
		gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
		gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
		gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)


		# top pad
		left_top_pad = nn.ZeroPad2d((0,2,0,2))
		gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

		center_top_pad = nn.ZeroPad2d((1,1,0,2))
		gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

		right_top_pad = nn.ZeroPad2d((2,0,0,2))
		gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

		# center pad
		left_center_pad = nn.ZeroPad2d((0,2,1,1))
		gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

		right_center_pad = nn.ZeroPad2d((2,0,1,1))
		gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

		# bottom pad
		left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
		gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

		center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
		gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

		right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
		gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

		gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
							 gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

		# normalize affinity using their abs sum
		gate_wb_abs = torch.abs(gate_wb)
		abs_weight = self.sum_conv(gate_wb_abs)

		gate_wb = torch.div(gate_wb, abs_weight)
		gate_sum = self.sum_conv(gate_wb)

		gate_sum = gate_sum.squeeze(1)
		gate_sum = gate_sum[:, :, 1:-1, 1:-1]

		return gate_wb, gate_sum


	def pad_blur_mask(self, blur_mask):
		# top pad
		left_top_pad = nn.ZeroPad2d((0,2,0,2))
		blur_mask_1 = left_top_pad(blur_mask).unsqueeze(1)
		center_top_pad = nn.ZeroPad2d((1,1,0,2))
		blur_mask_2 = center_top_pad(blur_mask).unsqueeze(1)
		right_top_pad = nn.ZeroPad2d((2,0,0,2))
		blur_mask_3 = right_top_pad(blur_mask).unsqueeze(1)

		# center pad
		left_center_pad = nn.ZeroPad2d((0,2,1,1))
		blur_mask_4 = left_center_pad(blur_mask).unsqueeze(1)
		right_center_pad = nn.ZeroPad2d((2,0,1,1))
		blur_mask_5 = right_center_pad(blur_mask).unsqueeze(1)

		# bottom pad
		left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
		blur_mask_6 = left_bottom_pad(blur_mask).unsqueeze(1)
		center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
		blur_mask_7 = center_bottom_pad(blur_mask).unsqueeze(1)
		right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
		blur_mask_8 = right_bottm_pad(blur_mask).unsqueeze(1)

		result_mask = torch.cat((blur_mask_1, blur_mask_2, blur_mask_3, blur_mask_4,
								  blur_mask_5, blur_mask_6, blur_mask_7, blur_mask_8), 1)
		return result_mask


	def normalize_gate(self, guidance):
		gate1_x1_g1 = guidance.narrow(1,0,1)
		gate1_x1_g2 = guidance.narrow(1,1,1)
		gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
		gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
		elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
		gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
		gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
		return gate1_x1_g1_cmb, gate1_x1_g2_cmb


	def max_of_4_tensor(self, element1, element2, element3, element4):
		max_element1_2 = torch.max(element1, element2)
		max_element3_4 = torch.max(element3, element4)
		return torch.max(max_element1_2, max_element3_4)

	def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
		max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
		max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
		return torch.max(max_element1_2, max_element3_4)

class Multi_Class_Affinity_Propagate(nn.Module):

	def __init__(self,
				 prop_time,
				 prop_kernel,
				 num_classes,
				 norm_type='8sum'):
		"""
		Inputs:
			prop_time: how many steps for CSPN to perform
			prop_kernel: the size of kernel (current only support 3x3)
			way to normalize affinity
				'8sum': normalize using 8 surrounding neighborhood
				'8sum_abs': normalization enforcing affinity to be positive
							This will lead the center affinity to be 0
		"""
		super(Multi_Class_Affinity_Propagate, self).__init__()
		self.num_classes = num_classes
		self.prop_time = prop_time
		self.prop_kernel = prop_kernel
		assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

		self.norm_type = norm_type
		assert norm_type in ['8sum', '8sum_abs']

		self.in_feature = 1
		self.out_feature = 1


	def forward(self, guidance, blur_mask, sparse_mask=None):
		#print("guidance: ", guidance.shape)
		#print("blur_mask: ", blur_mask.shape)
		# self.sum_conv = nn.Conv2d(in_channels=8,
		#                           out_channels=1,
		#                           kernel_size=(1, 1),
		#                           stride=1,
		#                           padding=0,
		#                           bias=False)
		# weight = torch.ones(1, 8, 1, 1)# .cuda()
		# self.sum_conv.weight = nn.Parameter(weight)
		# for param in self.sum_conv.parameters():
		#     param.requires_grad = False
		b, _, h, w = guidance.shape
		guidance = guidance.reshape(b,8,self.num_classes,h,w)
		# blur_mask = blur_mask.unsqueeze(1) # shape(b,1,num_classes,h,w)

		gate_wb, gate_sum = self.affinity_normalization(guidance)
		# (b,8,n,h+2,w+2), (b,n,h,w)

		# pad input and convert to 8 channel 3D features
		raw_mask_input = blur_mask

		#blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
		result_mask = blur_mask # (b,n,h,w)

		if sparse_mask is not None:
			sparse_mask = sparse_mask.sign()

		for i in range(self.prop_time):
			# one propagation
			# spn_kernel = self.prop_kernel
			result_mask = self.pad_blur_mask(result_mask) # (b,8,n,h+2,w+2)
			#print("gate_wb: ", gate_wb.shape)
			#print("result_mask: ", result_mask.shape)
			# neigbor_weighted_sum = self.sum_conv(gate_wb * result_mask)
			neigbor_weighted_sum = (gate_wb * result_mask).sum(dim=1) # (b,n,h+2,w+2)
			# neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
			neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1] # (b,n,h,w)
			result_mask = neigbor_weighted_sum

			if '8sum' in self.norm_type:
				result_mask = (1.0 - gate_sum) * raw_mask_input + result_mask
			else:
				raise ValueError('unknown norm %s' % self.norm_type)

			if sparse_mask is not None:
				result_mask = (1 - sparse_mask) * result_mask + sparse_mask * raw_mask_input

		return result_mask

	def affinity_normalization(self, guidance):

		# normalize features
		if 'abs' in self.norm_type:
			guidance = torch.abs(guidance)

		gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature) # (b,1,n,h,w)
		gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
		gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
		gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
		gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
		gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
		gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
		gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)


		# top pad
		left_top_pad = (0,2,0,2)
		gate1_wb_cmb = F.pad(gate1_wb_cmb, left_top_pad)# .unsqueeze(1)

		center_top_pad = (1,1,0,2)
		gate2_wb_cmb = F.pad(gate2_wb_cmb, center_top_pad)# .unsqueeze(1)

		right_top_pad = (2,0,0,2)
		gate3_wb_cmb = F.pad(gate3_wb_cmb, right_top_pad)# .unsqueeze(1)

		# center pad
		left_center_pad = (0,2,1,1)
		gate4_wb_cmb = F.pad(gate4_wb_cmb, left_center_pad)# .unsqueeze(1)

		right_center_pad = (2,0,1,1)
		gate5_wb_cmb = F.pad(gate5_wb_cmb, right_center_pad)# .unsqueeze(1)

		# bottom pad
		left_bottom_pad = (0,2,2,0)
		gate6_wb_cmb = F.pad(gate6_wb_cmb, left_bottom_pad)# .unsqueeze(1)

		center_bottom_pad = (1,1,2,0)
		gate7_wb_cmb = F.pad(gate7_wb_cmb, center_bottom_pad)# .unsqueeze(1)

		right_bottm_pad = (2,0,2,0)
		gate8_wb_cmb = F.pad(gate8_wb_cmb, right_bottm_pad)# .unsqueeze(1)

		gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
							 gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1) # (b,8,n,h+2,w+2)

		# normalize affinity using their abs sum
		gate_wb_abs = torch.abs(gate_wb)
		# abs_weight = self.sum_conv(gate_wb_abs)
		# print(gate_wb_abs.shape)
		abs_weight = gate_wb_abs.sum(dim=1, keepdim=True) # (b,1,n,h+2,w+2)

		gate_wb = torch.div(gate_wb, abs_weight)
		# gate_sum = self.sum_conv(gate_wb)
		gate_sum = gate_wb.sum(dim=1)  # (b,n,h+2,w+2)

		# gate_sum = gate_sum.squeeze(1)
		gate_sum = gate_sum[:, :, 1:-1, 1:-1] # (b,n,h,w)

		return gate_wb, gate_sum


	def pad_blur_mask(self, blur_mask):
		# top pad
		left_top_pad = nn.ZeroPad2d((0,2,0,2))
		blur_mask_1 = left_top_pad(blur_mask).unsqueeze(1) # (b,1,n,h+2,w+2)
		center_top_pad = nn.ZeroPad2d((1,1,0,2))
		blur_mask_2 = center_top_pad(blur_mask).unsqueeze(1)
		right_top_pad = nn.ZeroPad2d((2,0,0,2))
		blur_mask_3 = right_top_pad(blur_mask).unsqueeze(1)

		# center pad
		left_center_pad = nn.ZeroPad2d((0,2,1,1))
		blur_mask_4 = left_center_pad(blur_mask).unsqueeze(1)
		right_center_pad = nn.ZeroPad2d((2,0,1,1))
		blur_mask_5 = right_center_pad(blur_mask).unsqueeze(1)

		# bottom pad
		left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
		blur_mask_6 = left_bottom_pad(blur_mask).unsqueeze(1)
		center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
		blur_mask_7 = center_bottom_pad(blur_mask).unsqueeze(1)
		right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
		blur_mask_8 = right_bottm_pad(blur_mask).unsqueeze(1)

		result_mask = torch.cat((blur_mask_1, blur_mask_2, blur_mask_3, blur_mask_4,
								  blur_mask_5, blur_mask_6, blur_mask_7, blur_mask_8), 1)
		return result_mask # (b,8,n,h+2,w+2)


	def normalize_gate(self, guidance):
		gate1_x1_g1 = guidance.narrow(1,0,1)
		gate1_x1_g2 = guidance.narrow(1,1,1)
		gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
		gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
		elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
		gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
		gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
		return gate1_x1_g1_cmb, gate1_x1_g2_cmb


	def max_of_4_tensor(self, element1, element2, element3, element4):
		max_element1_2 = torch.max(element1, element2)
		max_element3_4 = torch.max(element3, element4)
		return torch.max(max_element1_2, max_element3_4)

	def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
		max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
		max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
		return torch.max(max_element1_2, max_element3_4)
