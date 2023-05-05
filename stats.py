from support import *

def STAT_TP_SIZE(model, label_folder, image_folder, sizes, code_type = 'qr', iou_threshold = 0.5): # ищем процент от gt который распознали без привязанности к правильности класса
  stat_tp_size_gen = np.zeros(len(sizes))
  stat_tp_size = np.zeros(len(sizes))
  counter_finded = 0
  counter_total = 0
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)

    if code_type == 'qr':
      for box in predicted_boxes:
        if box[5] == 0:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
              if box[4] == 0:
                se_gt_boxes.append(box)    

    if code_type == 'dm':
      for box in predicted_boxes:
        if box[5] == 1:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
        if box[4] == 1:
          se_gt_boxes.append(box)      

    se_predicted_boxes = np.array(se_predicted_boxes)
    se_gt_boxes = np.array(se_gt_boxes)
    if(len(se_gt_boxes) == 0):
      continue
    se_gt_boxes = se_gt_boxes[:,:4]
    counter_total = counter_total + len(se_gt_boxes)

    for boxA in se_gt_boxes:
      idx = get_idx(sizes, max(boxA[2]-boxA[0], boxA[3] - boxA[1]))
      stat_tp_size_gen[idx] = stat_tp_size_gen[idx] + 1
      for boxB in se_predicted_boxes:
        if(get_iou(boxA,boxB)) > iou_threshold:
          stat_tp_size[idx] = stat_tp_size[idx] + 1
          break
  return stat_tp_size,stat_tp_size_gen


def shift_box_stat(model, label_folder, image_folder, code_type = 'qr', iou_threshold = 0.5): # ищем процент от gt который распознали без привязанности к правильности класса

  shifts = []
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)

    if code_type == 'qr':
      for box in predicted_boxes:
        if box[5] == 0:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
              if box[4] == 0:
                se_gt_boxes.append(box)    

    if code_type == 'dm':
      for box in predicted_boxes:
        if box[5] == 1:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
        if box[4] == 1:
          se_gt_boxes.append(box)      

    se_predicted_boxes = np.array(se_predicted_boxes)
    se_gt_boxes = np.array(se_gt_boxes)
    if len(se_gt_boxes != 0):
      se_gt_boxes = se_gt_boxes[:,:4]

      for boxA in se_gt_boxes:
        for boxB in se_predicted_boxes:
          if(get_iou(boxA,boxB)) > iou_threshold:
            shifts.append(get_max_shift(boxA,boxB) / get_min_size(boxA))
            break

  return shifts


def get_tp(model, label_folder, image_folder, code_type = 'qr', iou_threshold = 0.5): # ищем процент от gt который распознали
  counter_finded = 0
  counter_total = 0
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)

    if code_type == 'qr':
      for box in predicted_boxes:
        if box[5] == 0:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
              if box[4] == 0:
                se_gt_boxes.append(box)    

      se_predicted_boxes = np.array(se_predicted_boxes)
      se_gt_boxes = np.array(se_gt_boxes)

    if code_type == 'dm':
      for box in predicted_boxes:
        if box[5] == 1:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
        if box[4] == 1:
          se_gt_boxes.append(box)      

      se_predicted_boxes = np.array(se_predicted_boxes)
      se_gt_boxes = np.array(se_gt_boxes)

    if(len(se_predicted_boxes) != 0):
      se_predicted_boxes = se_predicted_boxes[:,:4]
    else:
      counter_total = counter_total + len(se_gt_boxes)
      continue

    if(len(se_gt_boxes) == 0):
      continue
    se_gt_boxes = se_gt_boxes[:,:4]
    counter_total = counter_total + len(se_gt_boxes)
    for boxA in se_gt_boxes:
      find = False
      for boxB in se_predicted_boxes:
        if(get_iou(boxA,boxB)) > iou_threshold:
          find = True
          break
      if find == True:
        counter_finded = counter_finded + 1
  return float(counter_finded/counter_total)


def get_fp(model, label_folder, image_folder, code_type = 'qr', iou_threshold = 0.5): # ищем процент от gt который распознали
  counter_missed = 0
  counter_finded = 0
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)

    if code_type == 'qr':
      for box in predicted_boxes:
        if box[5] == 0:
          se_predicted_boxes.append(box)

    if code_type == 'dm':
      for box in predicted_boxes:
        if box[5] == 1:
          se_predicted_boxes.append(box)

    se_predicted_boxes = np.array(se_predicted_boxes)
    se_gt_boxes = np.array(gt_boxes)

    if(len(se_predicted_boxes) != 0):
      se_predicted_boxes = se_predicted_boxes[:,:4]
    else:
      continue

    if(len(se_gt_boxes) == 0):
        counter_missed = counter_missed + len(se_predicted_boxes)
        continue
        
    se_gt_boxes = se_gt_boxes[:,:4]
    counter_finded = counter_finded + len(se_predicted_boxes)

    for boxA in se_predicted_boxes:
      finded = False
      if not finded:
        for boxB in se_gt_boxes:
          if(get_iou(boxA,boxB)) > iou_threshold:
            finded = True
            break 
        if not finded:
          counter_missed = counter_missed + 1 
      else:
        continue

  return counter_missed


def look_at_misses(model, label_folder, image_folder, code_type = 'qr', iou_threshold = 0.5): # ищем процент от gt который распознали
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)

    if code_type == 'qr':
      for box in predicted_boxes:
        if box[5] == 0:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
              if box[4] == 0:
                se_gt_boxes.append(box)    

      se_predicted_boxes = np.array(se_predicted_boxes)
      se_gt_boxes = np.array(se_gt_boxes)

    if code_type == 'dm':
      for box in predicted_boxes:
        if box[5] == 1:
          se_predicted_boxes.append(box)
      for box in gt_boxes:
        if box[4] == 1:
          se_gt_boxes.append(box)      

      se_predicted_boxes = np.array(se_predicted_boxes)
      se_gt_boxes = np.array(se_gt_boxes)

    if(len(se_predicted_boxes) != 0):
      se_predicted_boxes = se_predicted_boxes[:,:4]
    else:
      if len(se_gt_boxes) != 0:
        dst_img = plot_bboxes(image, results[0].boxes.data, score=False)
        cv2.imwrite("results/images/wrong/" + code_type + "/" + path, dst_img)
      continue

    if(len(se_gt_boxes) == 0 and len(se_predicted_boxes) != 0):
      dst_img = plot_bboxes(image, results[0].boxes.data, score=False)
      cv2.imwrite("results/images/wrong/" + code_type + "/" + path, dst_img)
      continue
      
    se_gt_boxes = se_gt_boxes[:,:4]
    for boxA in se_gt_boxes:
      find = False
      for boxB in se_predicted_boxes:
        if(get_iou(boxA,boxB)) > iou_threshold:
          find = True
          break
      if not find:
          dst_img = plot_bboxes(image, results[0].boxes.data, score=False)
          cv2.imwrite("results/images/wrong/" + code_type + "/" + path, dst_img)

def get_swapped(model, label_folder, image_folder, iou_threshold = 0.5): # ищем процент от gt который распознали
  counter_missed = 0
  counter_total = 0
  for path in os.listdir(image_folder):
    se_predicted_boxes = []
    se_gt_boxes = []
    image = cv2.imread(image_folder + "/" + path)
    results = model.predict(image);
    predicted_boxes = np.array(results[0].boxes.data.cpu())
    predicted_boxes = predicted_boxes.astype("int32")
    gt_boxes = read_gt_boxes(label_folder + "/" + path[:-3] + "txt", image.shape)
  
    se_predicted_boxes = np.array(predicted_boxes)
    se_gt_boxes = np.array(gt_boxes)    
    counter_total = counter_total + len(se_gt_boxes)

    for boxA in se_gt_boxes:
      for boxB in se_predicted_boxes:
        if(get_iou(boxA,boxB)) > iou_threshold:
          if(boxB[5] != boxA[4]):
            counter_missed = counter_missed + 1
          
  return float(counter_missed/counter_total)