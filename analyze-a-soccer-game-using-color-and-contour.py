import numpy as np
import cv2

k=np.ones((55,77),np.uint8)

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def out(frame):

    lower,upper=[17, 15, 100], [50, 56, 200] #red
    mask1=detect_color(frame,lower,upper)
    lower,upper=[25, 146, 190], [96, 174, 250]#yellow
    mask2=detect_color(frame,lower,upper)
    mask=mask1+mask2
    #cv2.imshow('',cv2.resize(mask,(720,420)))

    output = cv2.bitwise_and(frame, frame, mask = mask)
    return output,mask

def detect_color(image,lower,upper):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        mask=cv2.dilate(mask,k)

        return mask

def contour(frame,mask):

    contours,h=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    org=frame
    text="UnKnow"
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,255,255), 5)
        if (x!=y):
            res=org[y:y+h,x:x+w]

            try:
                color=detect_team(res)
                if color != 'not_sure':
                    if color == 'red':
                        text = 'PERU'
                        cv2.rectangle(frame,(x-3,y-40),(x+w+3,y),(0,0,255),-1)
                    elif color == 'yellow':
                        text = 'AUS'
                        cv2.rectangle(frame,(x-3,y-40),(x+w+3,y),(0,255,255),-1)

                cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,2, (0,0,0), 3)
            except:pass
    return frame

def detect_team(image):

    mask1=detect_color(image,[17, 15, 100], [50, 56, 200]) #red
    mask2=detect_color(image,[25, 146, 190], [96, 174, 250]) #yellow

    output1 = cv2.bitwise_and(image, image, mask = mask1)
    output2 = cv2.bitwise_and(image, image, mask = mask2)

    color_pix1 = count_nonblack_np(output1)
    color_pix2 = count_nonblack_np(output2)
    if color_pix1!=0 and color_pix1>color_pix2 :
        return 'red'
    if color_pix2!=0 and color_pix2>color_pix1:
        return 'yellow'

    return 'not_sure'

cap=cv2.VideoCapture("C:\\Users\\Ahmad\\Desktop\\cv\\line\\soccer_realtime\\ft.mp4")

i=0
while cap.isOpened():
    _,frame = cap.read()
    if cv2.waitKey(1) == 13:
        break

    try:
        output,mask=out(frame) #find everything in yellow  or red
        output=contour(frame,mask) #crop imgs that is in yellow or red
        cv2.imshow("output",cv2.resize(output,(720,420)))

    except:
        pass

cap.release()
cv2.destroyAllWindows()
