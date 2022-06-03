import torch
import cv2
import numpy as np
import torchvision.transforms as tt
import model
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
from PIL import Image
import pathlib

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

device = 'cpu'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tif'])

# Specify the place to store the uploaded images
# Create the flask app
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


# Set the requirement for valid upload files
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Create the route for homepage upload.html
@app.route('/')
def upload_form():
    return render_template('upload.html')


# Take User uploaded images and perform prediction
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if '.tif' not in secure_filename(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join('static/uploads', filename))
            print(os.path.join('static/uploads', filename))
            im = Image.open(os.path.join('static/uploads', filename))
            im.save(os.path.join('static/uploads', filename.replace(str(pathlib.Path(filename).suffix), '.tif')), 'TIFF')
            filename = filename.replace(str(pathlib.Path(filename).suffix), '.tif')

        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join('static/uploads', filename))
        print("Success")

        cnn_model = model.Unet()
        cnn_model.load_state_dict(torch.load('brain-mri-unet-cpu.pth', map_location=torch.device('cpu')))
        image = cv2.imread(os.path.join('static/uploads', filename))
        image = cv2.resize(image, (256, 256))

        cnn_model.cpu()

        pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
        pred = tt.Normalize((0.0875, 0.0833, 0.0919), (0.1293, 0.1238, 0.1354))(pred)
        pred = cnn_model(pred)
        pred = pred.detach().cpu().numpy()[0, 0, :, :]

        pred_t = np.copy(pred)
        pred_t[np.nonzero(pred_t < 0.3)] = 0.0
        pred_t[np.nonzero(pred_t >= 0.3)] = 255.
        pred_t = pred_t.astype("uint8")
        cv2.imwrite(os.path.join('static/uploads', 'out_' + os.path.basename(filename).replace('.tif', '.jpg')), pred_t)
        mask = cv2.imread(os.path.join('static/uploads', 'out_' + os.path.basename(filename).replace('.tif', '.jpg')))
        ret, m_mask = cv2.threshold(mask[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image_and_mask = image.copy()
        image_and_mask[np.where(m_mask == 255)] = mask[np.where(m_mask == 255)]
        cv2.imwrite(os.path.join('static/uploads', 'out_' + os.path.basename(filename).replace('.tif', '.jpg')), image_and_mask)
        os.remove(os.path.join('static/uploads', os.path.basename(filename)))

        if np.all((pred_t == 0)):
            answer = 'No'
        else:
            answer = 'Yes'

        flash('The prediction is ' + answer)
        filename = os.path.join('out_' + os.path.basename(filename).replace('.tif', '.jpg'))
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, tif')
        return redirect(request.url)


# Create the route for the page after the prediction is made
# and show the images uploaded
@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static',
                    filename='uploads/' + filename, code=301))


# Create the route for Data description page data.html
@app.route('/data', methods=['GET', 'POST'])
def data_page():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('upload_form'))

    # show the form, it wasn't submitted
    return render_template('data.html')


# Create the route for the model page about.html
@app.route('/about', methods=['GET', 'POST'])
def about_page():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('upload_form'))
    return render_template('about.html')


if __name__ == "__main__":
    app.run(port=5085)