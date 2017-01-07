import cv2
import numpy as np
import os
import serial
import sqlite3
import time

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def get_serial_port():
    return "/dev/"+os.popen("dmesg | egrep ttyACM | cut -f3 -d: | tail -n1").read().strip()


ard = serial.Serial(get_serial_port(), baudrate=9600, timeout=3)

def open_gate():
    #TODO: send data to arduino to switch on led
    print('open gate')
    ard.write(b'1')
    print('on')
    time.sleep(5)
    return

def report_car():
    print('report car')
    ard.write(b'0')
    return

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, imgOriginalScene = cap.read()
        #imgOriginalScene = frame
        #cap.release()

        # attempt KNN training
        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

        # if KNN training was not successful
        if blnKNNTrainingSuccessful == False:
            print("\nerror: KNN traning was not successful\n")
            #return
        # end if

        # open image
        #imgOriginalScene = cv2.imread("1.png")

        # if image was not read successfully
        if imgOriginalScene is None:
            print("\nerror: image not read from file \n\n")
            #os.system("pause")
            #return
        # end if

        # show scene image
        #cv2.imshow("imgOriginalScene", imgOriginalScene)

        # detect plates
        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
        # detect chars in plates
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        # if no plates were found
        if len(listOfPossiblePlates) == 0:
            # inform user no plates were found
            #print("\nno license plates were detected\n")
            pass
        else:
            # if we get in here list of possible plates has at leat one plate
            # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            # suppose the plate with the most recognized chars (the first plate in
            # sorted by string length descending order) is the actual plate
            licPlate = listOfPossiblePlates[0]

            # show crop of plate and threshold of plate
            #cv2.imshow("imgPlate", licPlate.imgPlate)
            #cv2.imshow("imgThresh", licPlate.imgThresh)

            # if no chars were found in the plate
            if len(licPlate.strChars) == 0:
                #print("\nno characters were detected\n\n")
                #return
                continue
            # end if

            # draw red rectangle around plate
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

            #database program
            conn = sqlite3.connect('cars.db')
            cursor = conn.execute("SELECT ID FROM license_plates WHERE NUMBERS=? COLLATE NOCASE",(licPlate.strChars,))
            data = cursor.fetchall()

            if len(data) > 0:
                open_gate()
            else:
                report_car()

            # write license plate text to std out
            print("\nlicense plate read from image = " + licPlate.strChars + "\n")
            print("----------------------------------------")

            # write license plate text on the image
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

            # show scene image again
            #cv2.imshow("imgOriginalScene", imgOriginalScene)

            # write image out to file
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

        # end if else

    #cv2.waitkey(0)
    cv2.destroyAllWindows()
    return


# end main

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    # get 4 vertices of rotated rect
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


# end function


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    # this will be the center of the area the text will be written to
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    # choose a plain jane font
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    # base font scale on height of plate area
    fltFontScale = float(plateHeight) / 30.0
    # base font thickness on font scale
    intFontThickness = int(round(fltFontScale * 1.5))

    # call getTextSize
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    # make sure center is an integer
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    # the horizontal location of the text area is the same as the plate
    ptCenterOfTextAreaX = int(intPlateCenterX)

    # if the license plate is in the upper 3/4 of the image
    if intPlateCenterY < (sceneHeight * 0.75):
        # write the chars in below the plate
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        # else if the license plate is in the lower 1/4 of the image
        # write the chars in above the plate
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))
    # end if

    # unpack text size width and height
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    # based on the text area center, width, and height
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function
if __name__ == "__main__":
    main()
