import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def svm(img,mask):

# detect people in the image
    (rects, weights) = hog.detectMultiScale(mask, winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        res=img[y:y+h,x:x+w]
        color=detect_team(res)

        color_m= np.array([255,255,255], dtype = "uint8")
        if color != 'not_sure':
                    if color=='red':
                        color_m=[0,0,255]
                        text="PERU"
                    elif color=='yellow':
                        color_m=[0,255,255]
                        text="AUS"
        else:text="UnKnow"
        cv2.rectangle(img, (x, y), (x+w , y+h ),color_m, 3)
        cv2.rectangle(img, (x-2, y-25), (x+w+2 , y ),color_m, -1)
        cv2.putText(frame,text,(x+10,y-10),cv2.FONT_HERSHEY_PLAIN,1, (0,0,0), 2)
    return img

k=np.ones((55,55),np.uint8)

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def out(frame):
    lower,upper=[17, 15, 100], [50, 56, 200] #red
    mask1=detect_color(frame,lower,upper)

    lower,upper=[25, 146, 190], [96, 174, 250]#yellow
    mask2=detect_color(frame,lower,upper)
    mask=mask1+mask2
    output = cv2.bitwise_and(frame, frame, mask = mask)
    return output,mask

def detect_color(image,lower,upper):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        mask=cv2.dilate(mask,k,iterations=1)


        return mask

def detect_team(image):

    mask1=detect_color(image,[17, 15, 100], [50, 56, 200]) #red
    mask2=detect_color(image,[25, 146, 190], [96, 174, 250]) #yellow

    output1 = cv2.bitwise_and(image, image, mask = mask1)
    output2 = cv2.bitwise_and(image, image, mask = mask2)
    color_pix1 = count_nonblack_np(output1)
    color_pix2 = count_nonblack_np(output2)

    if color_pix1!=0 and color_pix1>color_pix2:
        return 'red'
    if color_pix2!=0 and color_pix2>color_pix1:
        return 'yellow'

    return 'not_sure'




cap=cv2.VideoCapture("C:\\Users\\Ahmad\\Desktop\\cv\\line\\soccer_realtime\\ft.mp4")

while True:
    _,frame = cap.read()
    org=frame
    if cv2.waitKey(33) == 13:
        break

    try:
        output,mask=out(frame) #find everything in yellow  or red
        output=svm(frame,output)

        cv2.imshow("output",cv2.resize(output,(720,440)))

    except:pass




cap.release()
