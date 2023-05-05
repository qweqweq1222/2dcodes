from PIL import Image
from google.colab.patches import cv2_imshow
from ultralytics import YOLO
from io import BytesIO
from google.colab import files

import numpy as np
import requests
import cv2
import torch
import torchvision
import os
import matplotlib.pyplot as plt