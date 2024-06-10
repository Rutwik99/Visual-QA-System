# Visual Question Answering Model Evaluation using VilT with Zero-Shot
# Team members: Jashwanth Sajja, Rohit Dhaipule and Rutwik Segireddy
# Description: This code loads a pre-trained VilT model and evaluates its performance on a subset of the VQA v2 dataset with 'yes/no' and numeric questions.
# Data frameworks used: 
#	- PyTorch for model training and evaluation
#	- PIL for image processing.
# Additional concepts used: 
#	- DataLoader for batch processing
#	- Regular Expressions for file handling.
#	- Transformers to implement ViLT
# System: Code was run on NVIDIA Quadro P5000 X2 using Ubuntu.

import json
import sys
import warnings
warnings.filterwarnings("ignore") 
import re
from typing import Optional
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltModel, ViltConfig


f = open('data/questions/v2_OpenEnded_mscoco_val2014_questions_yes_no.json')

# Return JSON object as dictionary
data_questions_val = json.load(f)
questions_val = data_questions_val['questions']

filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

def id_from_filename(filename: str) -> Optional[int]:
	"""Extract numeric ID from the filename."""
	match = filename_re.fullmatch(filename)
	if match is None:
		return None
	return int(match.group(1))

# root at which all images are stored
root_val = 'data/images/val2014'
file_names_val = [f for f in listdir(root_val) if isfile(join(root_val, f))]

filename_to_id_val = {root_val + "/" + file: id_from_filename(file) for file in file_names_val}
id_to_filename_val = {v:k for k,v in filename_to_id_val.items()}

# Read annotations
f = open('data/annotations/v2_mscoco_val2014_annotations_yes_no.json')

# Return JSON object as dictionary
data_annotations_val = json.load(f)
annotations_val = data_annotations_val['annotations']



config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

for annotation in annotations_val:
	annotation['labels'] = 1 if annotation['multiple_choice_answer'] == 'yes' else 0



class VQADataset(torch.utils.data.Dataset):
	"""Dataset class for VQA (v2)."""

	def __init__(self, questions, annotations, id_to_filename, processor):
		self.questions = questions
		self.annotations = annotations
		self.id_to_filename = id_to_filename
		self.processor = processor

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# get image + text
		annotation = self.annotations[idx]
		questions = self.questions[idx]
		if 'labels' not in annotation.keys():
			print(annotation)
			return None
		
		image = Image.open(self.id_to_filename[annotation['image_id']]).convert("RGB")
		

		text = questions['question']

		encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
		# remove batch dimension
		for k,v in encoding.items():
			encoding[k] = v.squeeze()
		  
		encoding["labels"] = annotation['labels']

		return encoding
	

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

dataset_val = VQADataset(questions=questions_val,
					 annotations=annotations_val,
					 id_to_filename=id_to_filename_val,
					 processor=processor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegressorModel(nn.Module):
	"""Regressor model for predicting numeric answers using VilT."""
	def __init__(self, config, num_labels):
		super().__init__()        
		num_images = -config.num_images

		self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")    
		self.classifier = nn.Sequential(
			nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images),
			nn.LayerNorm(config.hidden_size * num_images),
			nn.GELU(),
			nn.Linear(config.hidden_size * num_images, num_labels),
		)

	def forward(self, batch):
		x = self.model(**batch)
		x = torch.mean(x.last_hidden_state, dim=1)
		return self.classifier(x)

model = RegressorModel(config, 2)
model = model.to(device)



def collate_fn(batch):
	"""Collate function for batch processing."""
	input_ids = [item['input_ids'] for item in batch]
	pixel_values = [item['pixel_values'] for item in batch]
	attention_mask = [item['attention_mask'] for item in batch]
	token_type_ids = [item['token_type_ids'] for item in batch]
	labels = [[item['labels']] for item in batch]

	# create padded pixel values and corresponding pixel mask
	encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

	# create new batch
	batch = {}
	batch['input_ids'] = torch.stack(input_ids)
	batch['attention_mask'] = torch.stack(attention_mask)
	batch['token_type_ids'] = torch.stack(token_type_ids)
	batch['pixel_values'] = encoding['pixel_values']
	batch['pixel_mask'] = encoding['pixel_mask']
	batch['labels'] = torch.Tensor(labels)

	return batch

val_dataloader = DataLoader(dataset_val, collate_fn=collate_fn, batch_size=64, shuffle=False, num_workers=6)


model.eval()

correct = 0
total = 0

with torch.no_grad():
	
	for step, batch in enumerate(tqdm(val_dataloader)):
		# get the inputs;
		batch = {k:v.to(device) for k,v in batch.items()}
		labels = batch.pop("labels").to(device)
		# forward + backward
		outputs = model(batch)
		predicted = torch.argmax(outputs, dim=1)
		correct += (predicted.flatten().long() == labels.flatten().long()).sum().item()
		total += labels.size(0)
		
	print(correct/total)
