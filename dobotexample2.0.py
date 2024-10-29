import threading
import DoBotArm as dbt
import time
from serial.tools import list_ports
from Camera import *

# Initialisatie van de Dobot
ctrlDobot = dbt.DoBotArm("COM7", home=False)

# Lees de huidige positie van de robotarm in en stel die in als de home-positie
current_position = ctrlDobot.getCurrentPosition()  # Verondersteld dat deze methode (x, y, z) retourneert
homeX, homeY, homeZ = current_position  # Toewijzen van de uitgelezen waarden
print(f"Home-positie ingesteld op: X={homeX}, Y={homeY}, Z={homeZ}")

# correct_positions() retourneert een enkele lijst met coördinaten [x, y]
cX,cY = 100,100
matrix = correct_positions(cX,cY)  # Verkrijg de coördinatenlijst

print(matrix)  # Print de coördinatenlijst om te controleren wat erin zit

# x_waarde en y_waarde blokje 1
x_1 = matrix[0]  # Eerste element is de x-coördinaat
y_1 = matrix[1]  # Tweede element is de y-coördinaat
print(f"x_1: {x_1}")
print(f"y_1: {y_1}")

# Gebruik de coördinaten om de Dobot te verplaatsen
ctrlDobot.moveArmXYZ(x=x_1, y=y_1, z=30)


"""""
# Print de matrix netter
print("[")
for row in matrix:
    print(f" {row},")
print("]")
 
# x_waarde en y_waarde blokje 1
x_1 = matrix[0][0]  # Eerste rij, eerste kolom
y_1 = matrix[0][1]  # Eerste rij, tweede kolom
print(f"x_1: {x_1}")
print(f"y_1: {y_1}")
 
# x_waarde en y_waarde blokje 2
x_2 = matrix[1][0]  # Tweede rij, eerste kolom
y_2 = matrix[1][1]  # Tweede rij, tweede kolom
print(f"x_2: {x_2}")
print(f"y_2: {y_2}")


ctrlDobot.moveArmXYZ(x= 170, y= 50, z= 30)
ctrlDobot.moveArmXYZ(x= 170, y= 0, z= 30)
ctrlDobot.moveArmXYZ(x= 200, y= -100, z= 10)                     
ctrlDobot.moveArmXYZ(x= 200, y= -150, z= -20)
ctrlDobot.toggleSuction()
time.sleep(1)
ctrlDobot.moveArmXYZ(x= 200, y= -150, z= 60)
ctrlDobot.moveArmXYZ(x= 170, y= 50, z= 30)
ctrlDobot.toggleSuction()"""""


def move_to_camera_coords1(x_1, y_1):
    
    z_value = 30
# Beweeg de Dobot naar de x, y-coördinaten van de camera
    ctrlDobot.moveArmXYZ(x=x_1, y=y_1, z=z_value)


def move_to_camera_coords2(x_2, y_2):
    
    z_value = 30
# Beweeg de Dobot naar de x, y-coördinaten van de camera
    ctrlDobot.moveArmXYZ(x=x_2, y=y_2, z=z_value)
    
        
"""ctrlDobot.moveArmRelXYZ(0,0,30)
ctrlDobot.moveArmXYZ(x= 170, y= 0, z= 0)"""

# def SetConveyor(self, enabled, speed = 15000):


