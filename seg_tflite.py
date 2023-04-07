import numpy as np
import tensorflow as tf
import cv2

# https://www.tensorflow.org/lite/examples/object_detection/overview
# https://github.com/tensorflow/tensorflow/issues/51591
# https://pypi.org/project/googlenet-pytorch/

#model_path = "./object_detection_mobile_object_localizer_v1_1_default_1.tflite"
model_path = "static/models/20093fc4-4c4a-4976-9289-b81151d1e3ee.tflite"
#LABELS = ['Hartford', 'Zinfandel', 'Chambolle Musigny', 'Les Fuees', 'Carlisle']
LABELS = ['clean', 'dirty']
image_path= '../static/images/from_web.png'
THRESHOLD = 0.5
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype=np.uint8)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

img = cv2.imread(image_path)
original_image = img
print('Image shape ==>', original_image.shape)
img = cv2.resize(img, (input_height, input_width))

processed_img = tf.image.convert_image_dtype(img, tf.float32)

interpreter.set_tensor(input_details[0]['index'], [processed_img])
interpreter.invoke()




def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def draw_rect(results, image):
    h, w, c = image.shape
    for obj in results:
        y_min = int(max(1, (obj['bounding_box'][0] * h)))
        x_min = int(max(1, (obj['bounding_box'][1] * w)))
        y_max = int(min(h, (obj['bounding_box'][2] * h)))
        x_max = int(min(w, (obj['bounding_box'][3] * w)))

        class_id = int(obj['class_id'])
        label = "{}: {:.0f}%".format(LABELS[class_id], obj['score'] * 100)
        # color = [int(c) for c in COLORS[class_id]]
        color = (0, 0, 0)
        #y = y_min - 15 if y_min - 15 > 15 else y_min + 15

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (13, 13, 13), 2)
        cv2.putText(image, label, (x_min+15, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


# Get all outputs from the model
rects = get_output_tensor(interpreter, 0)
print()
print('rects ==>', rects)
classes = get_output_tensor(interpreter, 1)
print()
print('classes ==>', classes)
scores = get_output_tensor(interpreter, 2)
print()
print('scores ==>', scores)
count = int(get_output_tensor(interpreter, 3))
print()
print('count ==>', count)

results = []
for i in range(count):
    if classes[i] >= THRESHOLD:
        result = {'bounding_box': rects[i], 'class_id': scores[i], 'score': classes[i] }
        results.append(result)
print()
print('results ==>', results)

original_uint8 = draw_rect(results, original_image)
cv2.imshow("image", original_uint8)
cv2.waitKey(0)

