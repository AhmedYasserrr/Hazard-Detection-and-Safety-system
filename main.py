import torch
import numpy as np
import cv2

# constants
width = 1400
height = 720
mode = []
img = np.ones((height, width, 3), dtype=np.uint8)
img = 250 * img

# ellipse settings
angle = 0
startAngle = 0
endAngle = 360
color = (0, 0, 0)
thickness = 60
axesLength = (270, 30)

# font settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.9
font_color = (84, 184, 211)
font_color2 = (211, 184, 84)
font_thickness = 4

# list of objects to detect
detection_lst = []
while True:
    # Drawing the GUI for the system
    center_coordinates = (width // 2, height // 4)
    font_coordinates = (center_coordinates[0] - int(axesLength[0]//1.65), center_coordinates[1] + axesLength[1]//2)

    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, (84, 184, 211), thickness + 15)
    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, color, thickness)
    img = cv2.putText(img, '1-General Hazards Check',font_coordinates , font,
                        fontScale, font_color, font_thickness, cv2.LINE_AA)
    if 1 in mode:
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, font_color2, thickness + 15)
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, color, thickness)
        img = cv2.putText(img, '1-General Hazards Check', font_coordinates, font,
                          fontScale, font_color2, font_thickness, cv2.LINE_AA)

    center_coordinates = (width // 2, height // 2)
    font_coordinates = (center_coordinates[0] - axesLength[0]//2, center_coordinates[1] + axesLength[1]//2)

    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, (84, 184, 211), thickness + 15)
    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, color, thickness)
    img = cv2.putText(img, '2-Chemical PPE Check',font_coordinates , font,
                        fontScale, font_color, font_thickness, cv2.LINE_AA)
    if 2 in mode:
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, font_color2, thickness + 15)
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, color, thickness)
        img = cv2.putText(img, '2-Chemical PPE Check', font_coordinates, font,
                          fontScale, font_color2, font_thickness, cv2.LINE_AA)

    center_coordinates = (width // 2, height*3 // 4)
    font_coordinates = (center_coordinates[0] - axesLength[0]//2, center_coordinates[1] + axesLength[1]//2)
    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, (84, 184, 211), thickness + 15)
    img = cv2.ellipse(img, center_coordinates, axesLength,
                      angle, startAngle, endAngle, color, thickness)
    img = cv2.putText(img, '3-Physical PPE Check',font_coordinates , font,
                        fontScale, font_color, font_thickness, cv2.LINE_AA)
    if 3 in mode:
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, font_color2, thickness + 15)
        img = cv2.ellipse(img, center_coordinates, axesLength,
                          angle, startAngle, endAngle, color, thickness)
        img = cv2.putText(img, '3-Physical PPE Check', font_coordinates, font,
                          fontScale, font_color2, font_thickness, cv2.LINE_AA)

    img = cv2.putText(img, 'Hazard Detection and Safety system', (370, 50), font,
                        1.2 , font_color, 3, cv2.LINE_AA)
    img = cv2.putText(img, 'Note: press 1, 2, or 3 to select', (50, height - 50), font,
                        0.7 , font_color, 1, cv2.LINE_AA)
    img = cv2.putText(img, 'Press d to exit, if you are done selecting', (50, height - 20), font,
                        0.7 , font_color, 1, cv2.LINE_AA)
    cv2.imshow('Hazard Detection and Safety system', img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        mode.append(1)
        detection_lst.extend([0, 1, 15])

    elif cv2.waitKey(1) & 0xFF == ord('2'):
        mode.append(2)
        detection_lst.extend([10])

    elif cv2.waitKey(1) & 0xFF == ord('3'):
        mode.append(3)
        detection_lst.extend([3, 4])
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        img = cv2.putText(img, 'Model is processing.....', (width-350, height - 20), font,
                          0.7, font_color, 2, cv2.LINE_AA)
        cv2.imshow('Hazard Detection and Safety system', img)
        cv2.waitKey(0)
        break

# loading the trained custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt",
                       force_reload=True)
model.conf = 0.5  # NMS confidence threshold
model.classes = detection_lst  # filter objects to detect by class

# if model includes reflective vest, helmet, or gas mask
# new model to detect number of persons is loaded
if 3 or 4 or 10 in detection_lst:
    model2 = torch.hub.load('ultralytics/yolov5', "yolov5s", force_reload=True)
    model2.conf = 0.5  # NMS confidence threshold
    model2.classes = [0]  # person detection


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def s(num):
    if num < 2:
        return ""
    return "s"


# running the real-time model detection
cap = cv2.VideoCapture(0)
i = 0
cv2.destroyAllWindows()
while cap.isOpened():

    ret, frame = cap.read()
    # Make detections
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    df = results.xyxy[0].numpy()

    classes = []
    for i in range(len(df)):
        classes.append(df[i][5])

    if 3 or 4 or 10 in detection_lst:
        results2 = model2(np.squeeze(results.render()))
        df = results2.xyxy[0].numpy()
        classes2 = []
        for i in range(len(df)):
            classes2.append(df[i][5])
        num_persons = len(classes2)
        num_vests = classes.count(3.0)
        num_helmets = classes.count(4.0)
        num_masks = classes.count(10.0)
        results = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR)
        # results = rescale_frame(results, 50)
        h, w = results.shape[:2]
        if w > width:
            results = rescale_frame(results, 60)
        elif w < width//2:
            results = rescale_frame(results, 150)
        h, w = results.shape[:2]

        if 3 in detection_lst:
            if num_vests < num_persons:
                print(f"{num_persons} person with {num_vests} reflective vest")
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_vests} reflective vest{s(num_vests)}",
                                      (w // 20, h * 19 // 20), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_vests} reflective vest{s(num_vests)}",
                                      (w // 20, h * 19 // 20), font, 0.6, (0, 0, 220), 2, cv2.LINE_AA)

        if 4 in detection_lst:
            if num_helmets < num_persons:
                print(f"{num_persons} person with {num_helmets} helmet")
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_helmets} helmet{s(num_helmets)}",
                                      (w // 20, h * 17 // 20), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_helmets} helmet{s(num_helmets)}",
                                      (w // 20, h * 17 // 20), font, 0.6, (0, 0, 220), 2, cv2.LINE_AA)

        if 10 in detection_lst:
            if num_masks < num_persons:
                print(f"{num_persons} person with {num_masks} gas mask")
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_masks} gas mask{s(num_masks)}",
                                      (w // 20, h * 15 // 20), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                results = cv2.putText(results,
                                      f"{num_persons} person{s(num_persons)} with {num_masks} gas mask{s(num_masks)}",
                                      (w // 20, h * 15 // 20), font, 0.6, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.imshow('Hazard Detection and Safety system', results)
    else:
        results = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB)

    cv2.imshow('Hazard Detection and Safety system', results)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


