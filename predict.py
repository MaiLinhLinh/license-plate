from ultralytics import YOLO
from PIL import Image
model = YOLO('/teamspace/studios/this_studio/runs/detect/train/weights/best.pt')
results = model('/teamspace/studios/this_studio/594198054_1587258642466255_909828031103675193_n.jpg')

for r in results:
    print (r.boxes)
    im_arr = r.plot()
    im = Image.fromarray(im_arr[..., ::-1])
    im.show()
    im.save('result1.jpg')