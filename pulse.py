import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
print('Dlib loaded')

vstream = cv2.VideoCapture('stare.mp4')
frate = vstream.get(cv2.CAP_PROP_FPS)
nframes = int(vstream.get(cv2.CAP_PROP_FRAME_COUNT))
print('Stream loaded @ %i fps' % frate)

show = True
reds = np.zeros((nframes))  # red values of both cheeks
for i in tqdm(range(nframes)):
    _, image = vstream.read()
    if image is None: 
        break
    image = cv2.resize(image, None, fx=.5, fy=.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)  # detect faces in the grayscale image
    if len(rects) == 0: 
        continue

    landmarks = shape_to_np(predictor(gray, rects[0]))  # Get landmakrs from detected faces

    
    # https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-1024x825.jpg
    
    #IDEA: Would a different colorspace be better? could you just look at the hue?

    # Create contours for areas in the left and right cheeks (clockwise)
    lcheek_landmarks = (2,1,29,30)
    rcheek_landmarks = (15,14,30,29)
    cheeks = [np.array([landmarks[i] for i in lcheek_landmarks],dtype=np.int32), np.array([landmarks[i] for i in rcheek_landmarks],dtype=np.int32)]

    # Create mask and apply
    mask = np.zeros(image.shape[:2], dtype=np.int8)
    cv2.drawContours(mask, cheeks, -1, 1,-1)
    res = cv2.bitwise_and(image,image,mask = mask)
    reds[i] = res[mask==True].mean(axis=0)[2]

    if show:
        cv2.imshow('',res)
        key = cv2.waitKeyEx(0)
        if key == 8:  # delete/backspace
            exit()
        elif key == 13: # enter
            import code;code.interact(local=locals())
        elif key == 92: # pipe |\
            break
        elif key == 93: # right bracket ]
            show = False

    


    # nose_point = landmarks[30,:]
    # cheeks = landmarks[[15,3],:]  # left, right
    # mid_cheek_points = [tuple(((nose_point + cheek)/2).astype(int)) for cheek in cheeks]
    # p1, p2 = mid_cheek_points

    # # TODO: Is there a better colorspace to analyze this in?
    # reds[i, 0] = image[mid_cheek_points[0][::-1]][2]
    # reds[i, 1] = image[mid_cheek_points[1][::-1]][2]
    
    # show the output image with the face detections + facial landmarks
    # for i, l in enumerate(landmarks):
    #     cv2.circle(image, tuple(l), 1, (0,0,255), 2)
    #     cv2.putText(image, str(i), (l[0]+ 10, l[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # import code; code.interact(local=locals())


def autocorr(x):
	# https://stackoverflow.com/a/47369584
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]        
    if np.abs(r) > 0.5:
      print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    else: 
      print('Appears to be not autocorrelated')
    return r, lag, acorr

#Finds the fundamendamental freq of a series using autocorrelation
def fund_freq(x, sample_rate):
    pass #TODO: Autocorrelation

np.save('short_reds', reds)
autocorr(reds)

plt.plot(reds)
plt.show()