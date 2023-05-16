from stats import *


model = YOLO("best21042023I.pt")

src_prefix_labels = "test_dataset/labels/"
src_prefix_images = "test_dataset/images/"
sizes_boxes = []
sizes_imgs = []
for path in os.listdir(src_prefix_labels):
  img = cv2.imread(src_prefix_images + path[:-3] + "jpg")
  m = max(img.shape[0], img.shape[1])
  sizes_imgs.append(m)
  boxes = read_gt_boxes(src_prefix_labels + path, img.shape)
  for box in boxes:
    dx = box[2] - box[0]
    dy = box[3] - box[1]
    m = max(dx, dy)
    sizes_boxes.append(m)

sizes_imgs = np.array(sizes_imgs)
sizes_imgs = np.sort(sizes_imgs)

sizes_boxes = np.array(sizes_boxes)
sizes_boxes = np.sort(sizes_boxes)


fig = plt.figure(1)
plt.tight_layout()
plt.xlabel("сторона в пикселях")
plt.ylabel("количество")
plt.hist(sizes_imgs, bins = 50)

plt.title("максимальная сторона изображения")
plt.grid()
plt.savefig('results/plots/max_size_img.png')

plt.figure(2)
plt.xlabel("сторона в пикселях")
plt.ylabel("количество")
plt.hist(sizes_boxes, bins = 20)
plt.title("максимальная сторона коробки")
plt.grid()
plt.tight_layout()
plt.savefig('results/plots/max_size_box.png', bbox_inches="tight")

sizes = [20,50,100,200,500,100000]

stat_tp_size_qr, cp_qr = STAT_TP_SIZE(model, "test_dataset/labels", "test_dataset/images", sizes, code_type = "qr")
stat_tp_size_dm, cp_dm = STAT_TP_SIZE(model, "test_dataset/labels", "test_dataset/images", sizes, code_type = "dm")

fqr = open("results/stats_txt/size_distribution_qr.txt", "x")
fdm = open("results/stats_txt/size_distribution_dm.txt", "x")

fqr.write("QR decoded: ")
for stat in stat_tp_size_qr:
  fqr.write(str(int(stat)) + " ")
fqr.write("\n")
fqr.write("QR gt     : ")
for stat in cp_qr:
  fqr.write(str(int(stat)) + " ")

fqr.close()

fdm.write("DM decoded: ")
for stat in stat_tp_size_dm:
  fdm.write(str(int(stat)) + " ")
fdm.write("\n")
fdm.write("DM gt     : ")
for stat in cp_dm:
  fdm.write(str(int(stat)) + " ")

fdm.close()



shifts_qr = shift_box_stat(model, "test_dataset/labels/", "test_dataset/images/", "qr")
shifts_dm = shift_box_stat(model, "test_dataset/labels/", "test_dataset/images/", "dm")

plt.figure(3)
plt.xlabel("% от макс. стороны")
plt.ylabel("количество")
plt.hist(shifts_qr, bins = 30)
plt.title("распределение отступов QR")
plt.xticks(np.arange(0, max(np.array(shifts_qr)), 25))
plt.grid()
plt.tight_layout()
plt.savefig('results/plots/qr_shifts.png', bbox_inches="tight")

plt.figure(4)
plt.xlabel("% от макс. стороны")
plt.ylabel("количество")
plt.hist(shifts_dm, bins = 30)
plt.xticks(np.arange(0, max(np.array(shifts_dm)), 5))
plt.title("распределение отступов DM")
plt.grid()
plt.tight_layout()
plt.savefig('results/plots/dm_shifts.png', bbox_inches="tight")

tp_qr = get_tp(model, "test_dataset/labels/", "test_dataset/images/", "qr")
tp_dm = get_tp(model, "test_dataset/labels/", "test_dataset/images/", "dm")


fqr = open("results/stats_txt/tp_qr.txt", "x")
fdm = open("results/stats_txt/tp_dm.txt", "x")
fqr.write(str(tp_qr))
fdm.write(str(tp_dm))
fqr.close()
fdm.close()

fp_qr = get_fp(model, "test_dataset/labels/", "test_dataset/images/", "qr")
fp_dm = get_fp(model, "test_dataset/labels/", "test_dataset/images/", "dm")

fqr = open("results/stats_txt/fp_qr.txt", "x")
fdm = open("results/stats_txt/fp_dm.txt", "x")
fqr.write(str(fp_qr))
fdm.write(str(fp_dm))
fqr.close()
fdm.close()

qr_codes = 0
dm_codes = 0
for path in os.listdir("test_dataset/labels/"):
  img = cv2.imread("test_dataset/images/" + path[:-3] + "jpg")
  boxes = read_gt_boxes("test_dataset/labels/" + path, img.shape)
  for box in boxes:
    if box[4] == 0:
      qr_codes = qr_codes + 1 
    elif box[4] == 1:
      dm_codes = dm_codes +  1

print(fp_qr/qr_codes, fp_dm/dm_codes)

for path in os.listdir("test_dataset/images/"):
  img = cv2.imread("test_dataset/images/" + path)
  results = model.predict(img)
  box = read_gt_boxes("test_dataset/labels/" + path[:-3] + "txt", img.shape)
  dst_img = plot_bboxes(img, results[0].boxes.data, score=False)
  dst_img = cv2.rectangle(dst_img, (box[0,0],box[0,1]), (box[0,2],box[0,3]),(255,0,255), 7)
  cv2.imwrite("results/images/all/" + path, dst_img)

look_at_misses(model, "test_dataset/labels/", "test_dataset/images", code_type = 'qr', iou_threshold = 0.5)
look_at_misses(model, "test_dataset/labels/", "test_dataset/images", code_type = 'dm', iou_threshold = 0.5)

swapped = get_swapped(model, "test_dataset/labels/", "test_dataset/images")

f = open("results/stats_txt/misunderstood.txt", "x")
f.write(str(swapped))
f.close()
