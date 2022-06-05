import cv2
import numpy as np


class ObjectLocator:

    def __init__(self, shape, colour, frame):  #Initializing variables here
        self.shape = shape
        self.colour = colour
        self.job_done = False
        self.frame = frame
        self.real_life_coordinates = np.zeros(shape=(3,1))
    
    def main_loop(self):       #Main loop to run actions
        while(True):

            self.colour_detection(self.frame)
            self.shape_detection()
            cv2.imshow((f"Showing {self.colour} {self.shape}."), self.frame)
            cv2.waitKey(3500)
            
            if self.job_done == True:
                cv2.destroyAllWindows()
                break
        return False

    
    def colour_detection(self, frame):

        frame_to_mask = cv2.GaussianBlur(frame, (7,7), 1)               #Blurring image to reduce noise
        hsv = cv2.cvtColor(frame_to_mask, cv2.COLOR_BGR2HSV)
        
        
        lower_red = np.array([0,150,20])
        upper_red = np.array([5,255,255])

        lower_green = np.array([40,40,40])
        upper_green = np.array([70,255,255])

                                                #masking done here
        if self.colour =="green":
                self.mask = cv2.inRange(hsv, lower_green, upper_green)

        elif self.colour == "red":
                self.mask = cv2.inRange(hsv, lower_red, upper_red)

       


    def shape_detection(self):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):          #Iteration in contour
            area = cv2.contourArea(contour)

            if self.shape == "triangle":        #Getting shape according to trackbar
                shape_corners = 3
            elif self.shape == "rectangle":
                shape_corners = 4
            else:
                print("Shape must be triangle or rectangle...")
            
            if area > 1000:       #Filtering contour based on area of 1000 pixel
                
                if i == 0:
                    continue
                
                epsilon = 0.03*cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x , y, w, h= cv2.boundingRect(approx)       #Getting coordinates to calculate mid point of the object
                x_mid = int(x +w/3)
                y_mid = int(y + h/1.5)
                self.coordinates_for_h = np.array([[x_mid],[y_mid],[1]])    #appending the x,y values to np array to use in homography
                mid_coords = (x_mid, y_mid)
                _mid_coords = str(x_mid) + "," + str(y_mid)
                
                colour = (0,255,0)
                font = cv2.FONT_HERSHEY_COMPLEX
                
                if (len(approx) == shape_corners):   #Drawing contours based on the shape decision
                    cv2.drawContours(self.frame, [contour], 0, (0,0,0), 4)
                    cv2.putText(self.frame, _mid_coords, mid_coords, font, 1, colour, 1)
                    self.find_homography()
        


    def find_homography(self):
        pts_src = np.array([[202,36],[426,25],[202,365],[458,353]])             #Source points of plane from the frame 
        pts_dst = np.array([[0,0],[297,0],[0,210],[297,210]])                   #Destination points of the plane in real life

        h, status = cv2.findHomography(pts_src, pts_dst)
        self.real_life_coordinates = np.dot(h, self.coordinates_for_h)                          # Dot operation to find real value
        self.append_coordinates_totxt()
        

    def append_coordinates_totxt(self):
        f = open("FinalProduct/coordinates.txt", "a")
        f.write(self.colour + " " + self.shape + ": " + str(self.real_life_coordinates[0]) + " " + str(self.real_life_coordinates[1]))  #Appending the coordinate to txt file as requested
        f.write('\n')
        self.job_done = True
        







if __name__ == "__main__":
    
    flag = True
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap_copy= frame.copy()
    red_triangle = ObjectLocator(colour = "red", shape = "triangle", frame = cap_copy)
    green_triangle = ObjectLocator(colour = "green", shape = "triangle", frame = cap_copy)
    red_rectangle = ObjectLocator(colour = "red", shape = "rectangle", frame = cap_copy)
    green_rectangle = ObjectLocator(colour = "green", shape = "rectangle", frame = cap_copy)
    green_rectangle.main_loop()
    red_rectangle.main_loop()
    red_triangle.main_loop()
    green_triangle.main_loop()
