# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
import glob
from operator import itemgetter
import operator

app = Flask(__name__, template_folder='template')

dropzone = Dropzone(app)
# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

@app.route('/', methods=['GET', 'POST'])
def index():

	# set session for image results
	if "file_urls" not in session:
		session['file_urls'] = []
	# list to hold our uploaded image urls
	file_urls = session['file_urls']
	# handle image upload from Dropzone
	if request.method == 'POST':
		file_obj = request.files
		for f in file_obj:
			file = request.files.get(f)

			# save the file with to our photos folder
			filename = photos.save(
				file,
				name=file.filename
			)
			# append image urls
			file_urls.append(photos.url(filename))

		session['file_urls'] = file_urls
		return "uploading..."
	# return dropzone template on GET request
	return render_template('index.html')


@app.route('/results')
def results():

	# redirect to home if no images to display
	if "file_urls" not in session or session['file_urls'] == []:
		return redirect(url_for('index'))

	# set the file_urls and remove the session variable
	file_urls = session['file_urls']
	session.pop('file_urls', None)

	return render_template('results.html', file_urls=file_urls)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		file = request.files['file']
		extension = os.path.splitext(file.filename)[1]
		f_name = str(uuid.uuid4()) + extension
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
	return json.dumps({'filename':f_name})


@app.route('/viewpoints.html', methods=['GET', 'POST'])
def classify():
	print('do something here')
	imagePaths = sorted(list(paths.list_images('/uploads')))
	# loop over our testing images
	for imagePath in imagePaths:
		# load the image, resize it to a fixed 96 x 96 pixels (ignoring
		# aspect ratio), and then extract features from it
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (96, 96))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# load the network
		print("[INFO] loading network architecture and weights...")
		model = load_model('sorting.model')
		mlb = pickle.loads(open('mlb.pickle', "rb").read())

	# classify the image using our extracted features and pre-trained
	# neural network
		prob = model.predict(image)[0]
		dict1 = {}
		for (label, p) in zip(mlb.classes_, prob):
			dict1.update({label:p})
		print(dict1)
		dict2 = sorted(dict1, key=dict1.get, reverse=True)
		sorted_label = dict2[0]
	return "The image you uploaded has label - "+sorted_label

files = glob.glob('/uploads/*')
for f in files:
	os.remove(f)

	# when the button was clicked,
	# the code below will be execute.

if __name__ == '__main__':
	app.run()
