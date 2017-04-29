import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn import svm, datasets
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

cars = glob.glob('./trainingsamples/vehicles/*/*.png')
notcars = glob.glob('./trainingsamples/non-vehicles/*/*.png')

print (len(cars))
print (len(notcars))

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs, orient, 
                        pix_per_cell, cell_per_block, *,
                        vis=False, feature_vec=True,
                        cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        #Apply HoG
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], orient,
                                                pix_per_cell, cell_per_block,
                                                vis, feature_vec))
        hog_features = np.ravel(hog_features)  
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

def generate_scaler(cars, notcars, orient,
                        pix_per_cell, cell_per_block, *, 
                        vis=False, feature_vec=True,
                        cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):

    car_features = extract_features(cars, orient,
                        pix_per_cell, cell_per_block, 
                        vis=vis, feature_vec=feature_vec,
                        cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range)
    notcar_features = extract_features(notcars, orient,
                        pix_per_cell, cell_per_block, 
                        vis=vis, feature_vec=feature_vec,
                        cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range)

    feature_list = [car_features, notcar_features]

    # Create an array stack, NOTE: StandardScaler() expects np.float64
    X = np.vstack(feature_list).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define a labels vector based on features lists
    y = np.hstack((np.ones(len(car_features)), 
              np.zeros(len(notcar_features))))
    
    return scaled_X, y, X_scaler

def build_svc(scaled_X, y, *, kernel="linear", C=1, test_size=0.2):
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
                                        scaled_X, y, test_size=test_size, random_state=rand_state)
    
    #svc = svm.SVC(kernel=kernel, C=C, cache_size=1000, probability=True)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    return svc

def slide_window(img, window_list, *, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(150, 150), xy_overlap=(0.6, 0.6)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = img.shape[0]/2
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    # Return the list of windows
    return window_list

def window_features(image, orient, 
                        pix_per_cell, cell_per_block, *,
                        vis=False, feature_vec=True,
                        cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    #Apply HoG
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel], orient,
                                            pix_per_cell, cell_per_block,
                                            vis, feature_vec))
    hog_features = np.ravel(hog_features)  
    # Append the new feature vector to the features list
    features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

def search_windows(img, windows, clf, scaler, orient, 
                        pix_per_cell, cell_per_block, *,
                        vis=False, feature_vec=True,
                        cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), ystart = None, ystop=None, xstart=None):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using extract_features()
        features = window_features(test_img, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=vis, feature_vec=feature_vec,
                        cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range)
        #5) Scale extracted features to be fed to classifier
        features = np.float64(features)
        test_features = scaler.transform(features)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def add_array(heatmap_thresh, array_array):
    counter3 = 0
    while counter3 < 10: #NOTE: set to length of array_array - I know this is quick and dirty 
        heatmap_thresh += array_array[counter3]
        counter3 += 1
    return heatmap_thresh

def apply_threshold(heatmap_thresh, threshold=1):
    # Zero out pixels below the threshold
    heatmap_thresh[heatmap_thresh <= threshold] = 0
    # Return thresholded map
    return heatmap_thresh

def detect_cars(img, heatmap, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, *,
                        vis=False, feature_vec=False,
                        cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    window_list = []
    window_list = slide_window(img, window_list, xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    window_list = slide_window(img, window_list, xy_window=(96, 96), xy_overlap=(0.75, 0.75))
    window_list = slide_window(img, window_list, xy_window=(64, 64), xy_overlap=(0.75, 0.75))
    on = search_windows(img, window_list, svc, X_scaler, orient,
                        pix_per_cell, cell_per_block, vis=vis, feature_vec=feature_vec,
                        cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range)
    heatmap = add_heat(heatmap, on)
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        barea = ((np.max(nonzerox) - np.min(nonzerox)) * (np.max(nonzeroy) - np.min(nonzeroy)))
        print (barea)
        if barea > 3000:
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 8)
    # Return the image
    return img

# Define parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
vis = False
feature_vec = True
cspace = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)

scaled_X, y, X_scaler= generate_scaler(cars, notcars, orient,
                pix_per_cell, cell_per_block, 
                vis=vis, feature_vec=feature_vec,
                cspace=cspace, spatial_size=spatial_size,
                hist_bins=hist_bins, hist_range=hist_range)

try:
    print ("SVC found!")
    svc = joblib.load('svc1.pkl')
except FileNotFoundError:
    print ("Creating classifier...")
    svc = build_svc(scaled_X, y)
    joblib.dump(svc, 'svc1.pkl')

##init heatmap
img = mpimg.imread('./test_images/test1.jpg')
heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
heatmap_thresh = np.zeros_like(heatmap).astype(np.float)
heatmap_array = np.zeros([10,720,1280]).astype(np.float) #NOTE: Change add_array() and heatmap_array=heatmap for len
vid = cv2.VideoCapture('project_video.mp4')
counter = 0

while (True):
    ret, img = vid.read()
    heatmap = np.zeros_like(heatmap).astype(np.float)
    heatmap_thresh = np.zeros_like(heatmap).astype(np.float)
    correction = np.array([255, 255, 255]) #NOTE: to account for read-in format differences - remove if changing source
    img_corr = np.float32(np.divide(np.array(img), correction[None, None, :]))
    #find initial heatmap
    heatmap = detect_cars(img_corr, heatmap, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block, vis=vis, feature_vec=True,
                            cspace=cspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, hist_range=hist_range)
    heatmap = np.clip(heatmap, 0, 255)
    heatmap = apply_threshold(heatmap, 4) #soft vehicle detections
    heatmap_array[counter%10,:,:] = heatmap #update array_array - NOTE: update to array_array len for counter remainder
    heatmap_thresh = add_array(heatmap_thresh, heatmap_array) #collective heatmap
    heatmap_thresh = apply_threshold(heatmap_thresh, 30) #hard vehicle detections
    labels = label(heatmap_thresh) #label individual
    print('For Frame: ' + str(counter) + ', ' + str(labels[1]) + ' cars found')
    img = draw_labeled_bboxes(img, labels) #draw
    cv2.imwrite('./video_render/video_' + str(counter) + '.jpg', img)
    counter += 1
