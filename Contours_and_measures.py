import cv2


def showImage(image_name, image):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

def calculatePerimeter(object):
    perimeter = cv2.arcLength(object, True)
    perimeter = round(perimeter, 4)
    return perimeter

def calculateArea(object):
    area = cv2.contourArea(object)
    return area

def calculateAndDraw(contours):

    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            x1 = int(M['m10']/M['m00'])
            y1 = int(M['m01']/M['m00'])

        area = calculateArea(cnt)
        perimeter = calculatePerimeter(cnt)
        print(f'Area of contour {i+1}: ', area)
        print(f'Perimeter of contour {i+1}: ', perimeter)

        img_with_rectangles = drawRectangle(cnt)
        cv2.putText(img_with_rectangles, f'Area: {area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_with_rectangles, f'Perimeter: {perimeter}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    showImage("Medidas", image=img_with_rectangles)
    saveImage('Resources/measures.jpg', image_copy)


def drawRectangle(object):
    img_with_rectangle = cv2.drawContours(img, [object], -1, (0, 255, 255), 3)
    return img_with_rectangle


def saveImage(path_and_name, image):
    cv2.imwrite(path_and_name, image)


img = cv2.imread("Resources/img-2.png")
assert img is not None, "File not be read, check the path"

# convert to gray scacle
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply threasholding to convert grayscale to binary image
ret, thresh = cv2.threshold(gray, 40, 255, 0)

# find the contours on the binary image
contours, hierarchy = cv2.findContours(image=thresh, 
                                        mode=cv2.RETR_TREE, 
                                        method=cv2.CHAIN_APPROX_SIMPLE)

image_copy = img.copy()
cv2.drawContours(
    image=image_copy, 
    contours=contours, 
    contourIdx=-1, 
    color=(255, 255, 0), 
    thickness=2, 
    lineType=cv2.LINE_AA)


# Show image with contours
showImage('None aproximation', image_copy)

print("Number of objects detected: ", len(contours))
calculateAndDraw(contours=contours)


