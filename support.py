from libs import *

def read_gt_boxes(path, shape):
  boxes = []
  with open(path) as f:
      for line in f: # read rest of lines
          boxes.append([float(x) for x in line.split()])
  boxes = np.array(boxes)
  for box in boxes:
    box[1] = box[1] * shape[1]
    box[3] = box[3] * shape[1]
    box[2] = box[2] * shape[0]
    box[4] = box[4] * shape[0]
  boxes = boxes.astype("int32")
  ab_format = []
  for box in boxes:
    ab_format.append([box[1] - int(box[3] / 2), box[2] - int(box[4] / 2),box[1] + int(box[3] / 2), box[2] + int(box[4] / 2), box[0]])  
  ab_format = np.array(ab_format)
  return ab_format

def get_iou(bb1, bb2):

  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
  bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou

def box_label(image, box, prob, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label + "\n" + str(float('{:.3f}'.format(prob.item()))) , (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  if labels == []:
    labels = {0: u'__background__', 1: u'qr', 2: u'dm'}
  if colors == []:
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24)]
  
  for box in boxes:
    if score :
      label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])+1]
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        box_label(image, box, box[4], label, color)
    else:
      color = colors[int(box[-1])]
      box_label(image, box, box[4], label, color)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  try:
    import google.colab
    IN_COLAB = True
  except:
    IN_COLAB = False
  return image

def get_idx(sizes, current):
  for i, size in enumerate(sizes):
    if(current < size):
      return i
  return len(sizes) - 1

def get_max_shift(boxA, boxB):
  dx_left = abs(boxA[0] - boxB[0])
  dy_top  = abs(boxA[1] - boxB[1])
  dx_right = abs(boxA[2] - boxB[2])
  dy_bot = abs(boxA[3] - boxB[3])
  return max(dx_left, dx_right,dy_top, dy_bot)

def get_min_size(box):
  dx = abs(box[2] - box[0]) 
  dy = abs(box[3] - box[1])
  return min(dx,dy)
