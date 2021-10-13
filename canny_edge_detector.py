# Python program - Canny Edge Detection -  Lecture 20
  
# importing OpenCV(cv2) module
import cv2
import numpy as np
import math
  
# Save image in set directory
# Read RGB image
img = cv2.imread('valve.PNG') 

blur2 = [[1/16,2/16,1/16],
                 [2/16,4/16,2/16],
                  [1/16,2/16,1/16]]
        
blur = [[1/9,1/9,1/9],
               [1/9,1/9,1/9],
               [1/9,1/9,1/9]]

            
gx_sobel = [[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]]

gy_sobel = [[ 1, 2, 1],
                        [ 0, 0, 0],
                        [-1,-2,-1]]

def apply_blur(input, kernel):
    output = np.zeros((input.shape[0],input.shape[1]))
    
    for x in range(1,input.shape[0]-1):
        for y in range(1,input.shape[1]-1):
            for i in range(-1,2):
                for j in range(-1,2):
                    indx = x+i
                    indy = j+y
                    output[x,y] += input[indx,indy] * kernel[i+1][j+1]
    
    return output


def apply_sobel(input,Gx,Gy):
    #assumes a 2D matrix
    output = np.zeros((input.shape[0],input.shape[1]))
    for x in range(1,input.shape[0]-1):
        for y in range(1,input.shape[1]-1):
            pixel_x = 0 
            pixel_y = 0
            for i in range(-1,2):
                for j in range(-1,2):
                    indx = x+i
                    indy = j+y
                    pixel_x += input[indx,indy] * Gx[i+1][j+1]
                    pixel_y += input[indx,indy] * Gy[i+1][j+1]
                    
            val = math.sqrt(pixel_x * pixel_x + pixel_y * pixel_y)
            if val > 255: 
                val = 255
                
            output[x,y] = val
                    
                    
    return output
  
def apply_sup(input):
    output = np.zeros((input.shape[0],input.shape[1]))
    for x in range(1,input.shape[0]-1):
        for y in range(1,input.shape[1]-1):
            temp = np.zeros((3,3))
            for i in range(-1,2):
                for j in range(-1,2):
                    indx = x+i
                    indy = j+y
                    temp[i+1,j+1] = input[indx,indy]
            if np.amax(temp) == input[x,y]:
                output[x,y] = input[x,y]
            
    return output
    
def double_threshold(input):
    output = np.zeros((input.shape[0],input.shape[1]))
    for x in range(1,input.shape[0]-1):
        for y in range(1,input.shape[1]-1):
            if input[x,y] > 50:
                output[x,y] = 255

    return output

def apply_hysteresis(input):
    #assumes a 2D matrix
    output = np.zeros((input.shape[0],input.shape[1]))
    for x in range(1,input.shape[0]-1):
        for y in range(1,input.shape[1]-1):
            strong = False
            for i in range(-1,2):
                for j in range(-1,2):
                    indx = x+i
                    indy = j+y
                    if  input[indx,indy] == 255 and indx != x and indy !=y:
                       strong = True
            if strong:
                output[x,y] = input[x,y]
    return output

def main():
    # Output img with window name as 'image'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('image', gray) 
    
    print("Image dimensions: ",gray.shape,type(gray))
    
    #output_blur = apply_blur(gray,blur2)
    
    output_sobel = apply_sobel(gray,gx_sobel,gy_sobel)
    
    output_sup = apply_sup(output_sobel)
    
    output_threshold = double_threshold(output_sup)
    
    output_hyst = apply_hysteresis(output_threshold)
    
    cv2.imwrite('output.jpg', output_hyst)
    img2 = cv2.imread('output.jpg') 
    cv2.imshow('image', img2) 
    #print(output)
    # Maintain output window utill
    # user presses a key
    cv2.waitKey(0)        
  
    # Destroying present windows on screen
    cv2.destroyAllWindows() 


if __name__== "__main__":
    main()
