import cv2
import numpy as np
import math
from decimal import Decimal, getcontext
 
# Function to find the dominant color within a region of interest (ROI)
def find_dominant_color(roi):
    pixels = roi.reshape(-1, 3)
    pixel_count = pixels.shape[0]
    pixel_sum = np.sum(pixels, axis=0)
    dominant_color = (pixel_sum / pixel_count).astype(np.uint8)
    color = "Unknown"
    if(dominant_color[0]>dominant_color[1] and dominant_color[0]>dominant_color[2]):
        color = "Blue"
    elif(dominant_color[1]>dominant_color[0] and dominant_color[1]>dominant_color[2]):
        color = "Green"
    elif(dominant_color[2]>dominant_color[0] and dominant_color[2]>dominant_color[0]):
        color = "Red"
   
    return color
 
# x, y, w, h = 243, 158, 222, 223 or 222, 198, 226, 226 or  149, 165, 221, 219 or 144, 166, 220, 217
x, y, w, h = 149, 165, 221, 219
x, y, w, h = x+10, y+10, w-20, h-20
 
# Posities
"""
DobotPosA = [0, 0]
DobotPosB = [10,10]
CameraPosA = [x, y]
CameraPosB = [x+w, y+h]
"""
DobotPosA = [-150,-275]
DobotPosB = [-113, -202]
CameraPosA = [264, 242]
CameraPosB = [340, 343]
 
# Bereken lengtes
Dobotlengte = math.sqrt((DobotPosB[0] - DobotPosA[0])**2 + (DobotPosB[1] - DobotPosA[1])**2)
Cameralengte = math.sqrt((CameraPosB[0] - CameraPosA[0])**2 + (CameraPosB[1] - CameraPosA[1])**2)
 
# Bereken rekfactor
RekFactor = Dobotlengte / Cameralengte
 
# Rekmatrix
RekMatrix = [[RekFactor, 0],
             [0, RekFactor]]
 
# Posities na rek
CameraPosRekA = [(RekMatrix[0][0] * CameraPosA[0] + RekMatrix[0][1] * CameraPosA[1]),
                 (RekMatrix[1][0] * CameraPosA[0] + RekMatrix[1][1] * CameraPosA[1])]
CameraPosRekB = [(RekMatrix[0][0] * CameraPosB[0] + RekMatrix[0][1] * CameraPosB[1]),
                 (RekMatrix[1][0] * CameraPosB[0] + RekMatrix[1][1] * CameraPosB[1])]
 
# Bereken hoeken x-as
DobotHoek = math.atan((DobotPosB[1] - DobotPosA[1])/(DobotPosB[0] - DobotPosA[0]))
CameraHoek = math.atan((CameraPosB[1] - CameraPosA[1])/(CameraPosB[0] - CameraPosA[0]))
 
# Bereken rotatiehoek
RotatieHoek = DobotHoek - CameraHoek
 
# Rotatiematrix
RotatieMatrix = [[math.cos(RotatieHoek), -math.sin(RotatieHoek)],
                 [math.sin(RotatieHoek), math.cos(RotatieHoek)]]
 
# Posities na rek en rotatie
CameraPosRekRotA = [(RotatieMatrix[0][0] * CameraPosRekA[0] + RotatieMatrix[0][1] * CameraPosRekA[1]),
                    (RotatieMatrix[1][0] * CameraPosRekA[0] + RotatieMatrix[1][1] * CameraPosRekA[1])]
CameraPosRekRotB = [(RotatieMatrix[0][0] * CameraPosRekB[0] + RotatieMatrix[0][1] * CameraPosRekB[1]),
                    (RotatieMatrix[1][0] * CameraPosRekB[0] + RotatieMatrix[1][1] * CameraPosRekB[1])]
 
# Berekenen translatievector
TranslatieVectorA = [DobotPosA[0] - CameraPosRekRotA[0], DobotPosA[1] - CameraPosRekRotA[1]]
TranslatieVectorB = [DobotPosB[0] - CameraPosRekRotB[0], DobotPosB[1] - CameraPosRekRotB[1]]
TranslatieVector = TranslatieVectorA
 
 
# Gekalibreerde posities
KalPosA = [(CameraPosRekRotA[0] + TranslatieVector[0]), (CameraPosRekRotA[1] + TranslatieVector[1])]
KalPosB = [(CameraPosRekRotB[0] + TranslatieVector[0]), (CameraPosRekRotB[1] + TranslatieVector[1])]
 
def correct_positions(cX,cY):
    #Positie
    ObjectPos = [cX,cY]
 
    #Positie na rek
    ObjectPosRek = [(RekMatrix[0][0] * ObjectPos[0] + RekMatrix[0][1] * ObjectPos[1]),
                    (RekMatrix[1][0] * ObjectPos[0] + RekMatrix[1][1] * ObjectPos[1])]
 
    #Positie na rek en rotatie
    ObjectPosRekRot = [(RotatieMatrix[0][0] * ObjectPosRek[0] + RotatieMatrix[0][1] * ObjectPosRek[1]),
                    (RotatieMatrix[1][0] * ObjectPosRek[0] + RotatieMatrix[1][1] * ObjectPosRek[1])]
 
    #gekalibreerde positie
    KalObjectPos = [(ObjectPosRekRot[0] + TranslatieVector[0]), (ObjectPosRekRot[1] + TranslatieVector[1])]
    KalObjectPos = [round(KalObjectPos[0], 2), round(KalObjectPos[1], 2)]
    return KalObjectPos
 
# Initialize the video capture from the USB camera (usually 0 for the default camera)
cap = cv2.VideoCapture(1)
 
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
 
    if not ret:
        break
 
    # Define the region of interest (ROI) for cropping
     
    cropped_frame = frame[y:y+h, x:x+w]
 
    # # Convert the frame to grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
   
    # # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # # Use Canny edge detection to find edges in the frame
    edges = cv2.Canny(blurred, 50, 150)
 
    # # Find contours in the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # # Loop over the detected contours
    for contour in contours:
    #     # Approximate the shape by a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
 
    #     # Calculate the number of vertices of the polygon
        num_vertices = len(approx)
 
    #     # Determine the shape based on the number of vertices
        shape = "Unknown"
        if num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            shape = "Square"
        elif num_vertices >= 8:
            shape = "Circle"
 
    #     # Get the position of the shape
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"])+x)
            cY = int((M["m01"] / M["m00"])+y)
        else:
            cX, cY = -1, -1  # Handle the case where no valid shape is found
 
    #     # Extract the ROI within the contour
        roi = frame[cY - 50:cY + 50, cX - 50:cX + 50]
    #     # Find the dominant color within the ROI
        dominant_color = find_dominant_color(roi)
 
    #     # Calculate positions
        positions = correct_positions(cX,cY)
 
    #     # Display the color, shape, and position on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.rectangle(frame, (cX - 25, cY - 25), (cX + 25, cY + 25), (0, 255, 0), 1)
        cv2.putText(frame, f"{dominant_color}", (cX - 25, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{shape}", (cX - 25, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"[{cX}, {cY}]", (cX - 25, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{positions}", (cX - 25, cY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
    # # Display the resulting frame
    cv2.imshow("Detected Shapes", frame)
 
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()