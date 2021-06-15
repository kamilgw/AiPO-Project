import numpy as np
import cv2 as cv

VIDEO_PATH = 'test_videos/input.mp4'
MODEL_LABELS = 'yolov3-coco/coco-labels'
MODEL_PATH = 'yolov3-coco/yolov3.cfg'
MODEL_WEIGHTS = 'yolov3-coco/yolov3.weights'


def draw_labels_and_boxes(img, boxes, classids, idxs, colors, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classids[i]]]

            if labels[classids[i]] == 'person':
                number = len(idxs)
            counter = "Number of People in Frame:{}".format(number)
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, counter, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            text = "{}".format(labels[classids[i]])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            if confidence > tconf and classid == 0:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids


def infer_image(net, layer_names, height, width, img, colors, labels,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    if infer:
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(layer_names)
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, 0.5)
        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise Exception('[ERROR] Required variables are set to None before drawing boxes on images.')

    img = draw_labels_and_boxes(img, boxes, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs


if __name__ == '__main__':

    labels = open(MODEL_LABELS).read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    net = cv.dnn.readNetFromDarknet(MODEL_PATH, MODEL_WEIGHTS)

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    try:
        vid = cv.VideoCapture(VIDEO_PATH)
        height, width = None, None
        writer = None
    except cv.error:
        print('Video cannot be loaded! Please check the path provided!')
    except Exception as e:
        print(e)

    finally:
        while True:
            grabbed, frame = vid.read()

            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels)

            cv.imshow('Frame', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        vid.release()

    vid.release()
    cv.destroyAllWindows()
