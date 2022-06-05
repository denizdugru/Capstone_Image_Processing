from cv2 import VideoCapture
import numpy as np
import cv2
import random



def main():

    cap = cv2.VideoCapture(0)
    def empty (a):
        pass

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    cv2.createTrackbar("Colour", "Parameters", 0, 1, empty)
    cv2.createTrackbar("Area", "Parameters", 1000, 30000, empty)
    cv2.createTrackbar("Shape", "Parameters", 0, 1, empty)

    while(True):
        success, frame = cap.read()
        frame_blur = cv2.GaussianBlur(frame, (7,7), 1)
        cap_copy= frame.copy()
        masked_frame = colour_detect(frame_blur)
        final_frame = shape_detection(masked_frame, cap_copy)
        cv2.imshow("Final", final_frame)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
            

 

def colour_detect(frame_blur,  debug_mode=None, colour_mask=None):

    
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV) 
    
    lower_red = np.array([0,150,20])
    upper_red = np.array([5,255,255])

    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])
    
    choice_red = [lower_red, upper_red]
    choice_green = [lower_green, upper_green]
    choices =  [choice_green, choice_red]
    temp_num = cv2.getTrackbarPos("Colour", "Parameters")
    random.seed(temp_num)
    temp_colour_mask = random.choice(choices)

    if debug_mode == "manual":
        if colour_mask =="green":
            mask = cv2.inRange(hsv, lower_green, upper_green)
            res = cv2.bitwise_and(frame_blur, frame_blur, mask= mask)    

        elif colour_mask == "red":
            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_and(frame_blur, frame_blur, mask= mask)
        
    elif debug_mode == None:
        mask = cv2.inRange(hsv, temp_colour_mask[0], temp_colour_mask[1])
        res = cv2.bitwise_and(frame_blur, frame_blur, mask= mask)

        
    cv2.imshow('mask', mask)
    return mask
    

def shape_detection(frame, cap_copy):

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        temp_num = cv2.getTrackbarPos("Shape", "Parameters")
        if temp_num == 1:
            shape = 3
        elif temp_num ==0:
            shape = 4
        else:
            pass
        if area > areaMin:
            if i == 0:
                continue
            epsilon = 0.03*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x , y, w, h= cv2.boundingRect(approx)
            x_mid = int(x +w/3)
            y_mid = int(y + h/1.5)

            coords = (x_mid, y_mid)
            colour = (0,255,0)
            font = cv2.FONT_HERSHEY_COMPLEX
            if (len(approx) == shape):
                cv2.drawContours(cap_copy, [contour], 0, (0,0,0), 4)
                n = approx.ravel()
                i = 0

                for j in n:
                    if(i%2 == 0):
                        x = n[i]
                        y = n[i+1]
                        coordinates = str(x) + " " + str(y)
                    else:
                        pass
                    
                    cv2.putText(cap_copy, coordinates, (x,y), font, 1, colour, 1)

                    i += 1
            
    cap_copy = cap_copy[0:600,0:600]
    return cap_copy
    

main()

