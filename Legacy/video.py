import cv2
import os

"""
be sure your images are nicely numbered.

images with nice, chronologic numbers :
    image1
    image2
    image3
    ...
    image256

images that wont work :
    image1
    zzt
    imge6
    img2
    imageimagetest42
"""


##### The folder in which your images are
image_folder = 'prof'
##### The name you want to give to your video
video_name = 'video.avi'


##### read the images
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")]) ### get the name of all files ending in png, ordered by name

##### Opening the images
frame = cv2.imread(os.path.join(image_folder, images[0]))

##### Getting their shape to define the video shape
height, width, layers = frame.shape

##### define the video
fps=15
video = cv2.VideoWriter(video_name, 0, fps, (width,height))

##### write images
for image in images:
    video.write(frame)

##### close the program, NEEDED
cv2.destroyAllWindows()
video.release()
