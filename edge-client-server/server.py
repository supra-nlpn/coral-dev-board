from flask import Flask, request, jsonify
from PIL import Image
import io
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from pycoral.utils import dataset
import numpy as np
from periphery import GPIO

interpreter = make_interpreter('mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
interpreter.allocate_tensors()
labels = dataset.read_label_file('inat_bird_labels.txt')

app = Flask(__name__)

def trigger():
    trigger_gpio = GPIO("/dev/gpiochip2", 9, "out") #physical-pin 16
    trigger_gpio.write(True)
    trigger_gpio.write(False)
    trigger_gpio.close()

@app.route('/predict', methods=['POST'])
def predict():
    trigger() # Trigger the GPIO pin to indicate a new request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    img = Image.open(image_file.stream).convert('RGB').resize((224, 224))
    trigger() # Trigger the GPIO pin to indicate image processing start
    common.set_input(interpreter, img)
    interpreter.invoke()
    trigger() # Trigger the GPIO pin to indicate inference completion
    result = classify.get_classes(interpreter, top_k=1)[0]
    trigger() # Trigger the GPIO pin to indicate response ready
    response = {
        'label': labels.get(int(result.id), 'unknown'),  # cast to int
        'class_id': int(result.id),                     # cast to int
        'score': float(result.score)                    # cast to float
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
