import os
import glob
import math
import matplotlib.pyplot as plt
import json, string
import torch
import pickle
import numpy
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.datasets as datasets
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class vgg16_bn_nc():
    def __init__(self):
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.classifier = None

    def forward(self, x):
        x = self.vgg16_bn.features(x)
        x = x.view(x.size(0), -1)
        return x

class custom_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(custom_classifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        return self.seq.forward(x)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def process_images(network, vggnet, img_path, transform, use_gpu=False):
    results = []
    data = datasets.ImageFolder(root=img_path, transform=transform)
    if use_gpu:
        network = network.cuda()
        vggnet.vgg16_bn = vggnet.vgg16_bn.cuda()
    network.eval()
    for (i, (inputs, _)) in enumerate(data):
            
            inputs = inputs.unsqueeze(0)
            inputs = Variable(inputs)
            if use_gpu:
                inputs = inputs.cuda()
            
            # Forward pass:
            vgg_out = vggnet.forward(inputs)
            outputs = network(vgg_out)
            results.append(max_index(outputs))
            
    return results

def max_index(tensor):
    # returns index of the larger value, expects a 1x2 tensor
    return 0 if tensor.data[0][0] > tensor.data[0][1] else 1

def img_to_output(output):
    # for debugging/verification purposes 
    for (i, n) in enumerate(output):
        print i+2, n

def identify_clips(output, offset=1):
    # pulls out continuous 1s from the output. Also allows a 0 if it's surrounded by 1s.
    # offset is applied to match indexing with image numbers 
    # returns a list of tuples, where the tuple = (clip_start_image_index, clip_end_image_index)

    clips = []
    start = -1
    for (i, n) in enumerate(output):
        if start == -1 and n == 1:
            start = i
        elif start != -1 and n == 0 and (i+1 < len(output) and output[i+1] == 0):
            clips.append((start+offset, i+offset))
            start = -1
    return clips

def filter_highlights(sorted_clips, p=0.5):
    num_clips = int(math.ceil(len(sorted_clips)*p))
    highlights = sorted_clips[:num_clips]
    highlights.sort(key=lambda x: x[0][0]) # return in sorted order by start index
    return highlights

def ts(i): return i * 1000

def identify_highlights(video_path, clips):
    sound = AudioSegment.from_file(video_path)
    video_length = len(sound)
    begin_offset = -2
    end_offset = 3
    l = []

    for clip in clips:
        end_i = clip[1]
        subsegment = sound[ts(end_i+begin_offset):min(ts(end_i+end_offset), video_length)]
        l.append((clip, subsegment.max))

    l.sort(key=lambda x: x[1], reverse=True)
    highlights = filter_highlights(l, p=0.5)
    return highlights

def create_highlight_video(video_path, highlights):
    video = VideoFileClip(video_path)
    highlight_clips = [ 
        VideoFileClip(video_path).subclip(highlights[i][0][0], highlights[i][0][1]) 
        for i in range(len(highlights))
    ]   
    final_clip = concatenate_videoclips(highlight_clips)
    final_clip.write_videofile("highlights.mp4", codec='libx264', audio=True)


video_path = './data/vid.mp4' # Path where the video is located
offset = 1
size = (224, 224)
imgTransform = transforms.Compose([
    transforms.Scale(size),
    transforms.ToTensor()
])

vgg = vgg16_bn_nc()
network = custom_classifier()
network.load_state_dict(torch.load('highlights.pt'))

output = process_images(network, vgg, "./data", imgTransform, use_gpu=True)
clips = identify_clips(output, offset)
highlights = identify_highlights(video_path, clips)
create_highlight_video(video_path, highlights)

