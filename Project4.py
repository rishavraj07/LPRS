# Automatic Number Plate Detection from given Image(passed)
import cv2
import pytesseract
import numpy as np

# Implementing OCR(Optical Character Recognition) on the cropped number plate.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

cascade= cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
states={"AN":"Andaman and Nicobar","AP":"Andhra Pradesh","AR":"Arunachal Pradesh","AS":"Assam","BR":"Bihar","CH":"Chandigarh","DN":"Dadra and Nagar Haveli","DD":"Daman and Diu","DL":"Delhi","GA":"Goa","GJ":"Gujarat",
"HR":"Haryana","HP":"Himachal Pradesh","JK":"Jammu and Kashmir","KA":"Karnataka","KL":"Kerala","LD":"Lakshadweep","MP":"Madhya Pradesh","MH":"Maharashtra","MN":"Manipur","ML":"Meghalaya","MZ":"Mizoram","NL":"Nagaland","OD":"Odissa","PY":"Pondicherry","PN":"Punjab","RJ":"Rajasthan","SK":"Sikkim","TN":"TamilNadu","TR":"Tripura","UP":"Uttar Pradesh", "WB":"West Bengal","CG":"Chhattisgarh","TS":"Telangana","JH":"Jharkhand","UK":"Uttarakhand"}

img = cv2.imread("Resources/numplte.jpg")## Reading Image
# Converting into Gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Detecting plate
nplate = cascade.detectMultiScale(gray,1.1,4)
count=0
for (x,y,w,h) in nplate:
        # Crop a portion of plate
        a,b = (int(0.025*img.shape[0]), int(0.025*img.shape[1]))

        plate = img[y+a:y+h-a, x+b:x+w-b, :]
        # make image more darker to identify the LPR
        ## iMAGE PROCESSING
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        # Feed Image to OCR engine
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        print(read)
        stat = read[0:2]
        try:
        # Fetch the State information
            print('Car Belongs to',states[stat])
        except:
            print('State not recognised!!')
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y),(51,51,255) , -1)
        cv2.putText(img,read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('PLate',plate)
        # Save & display result image
        cv2.imwrite('Resources/Project4_Output/plate.jpg', plate)

cv2.imshow("Result", img)
cv2.imwrite('Resources/Project4_Output/result.jpg',img)
cv2.waitKey(0)
