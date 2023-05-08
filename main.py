import os
import cv2
from PIL import Image
import imagehash
import subprocess
import torchvision
import streamlit as st


import yolov7
#from .detect import *

def count_vehicle_detections(video_path, labels_folder):
    # Load video file
    video = cv2.VideoCapture(video_path)

    # Get number of frames in video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize dictionary to count vehicle detections
    detections_dict = {'Bike':3, 'Car':1,'Heavy truck':2}
    # Loop through each frame and count vehicle detections
    for i in range(frame_count):
        # Load label file for current frame
        label_path = os.path.join(labels_folder, f'frame{i}.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            for label in labels:
                label = label.strip().split(' ')
                if label[0] in detections_dict.keys():
                    detections_dict[label[0]] += 1

    # Release video file
    video.release()

    return detections_dict






def main():
    st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)
    st.title("Vehicle Detection App")

    # Get video file from user input
    video_file = st.file_uploader("Upload video file", type=["mp4"])

    if video_file is not None:
        # Save video file to disk
        video_path = 'video.mp4'
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Load video file
        video = cv2.VideoCapture(video_path)


        # Create a folder to store the frames
        folder = 'frames'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create a folder to store the unique frames
        unique_folder = 'unique_frames'
        if not os.path.exists(unique_folder):
            os.makedirs(unique_folder)

        # Loop through each frame
        success, image = video.read()
        count = 0
        hashes = set()
        unique_count = 0
        while success:
            # Save the frame in the folder
            filename = os.path.join(folder, f'frame{count}.jpg')
            cv2.imwrite(filename, image)

            # Calculate image hash
            hash = str(imagehash.average_hash(Image.open(filename)))

            # Check if hash is unique
            if hash not in hashes:
                # Add hash to set of hashes
                hashes.add(hash)
                # Increment count of unique frames
                unique_count += 1
                # Save the unique frame to the unique folder
                unique_filename = os.path.join(unique_folder, f'unique_frame{unique_count}.jpg')
                cv2.imwrite(unique_filename, image)

            # Read the next frame
            success, image = video.read()
            count += 1

        print(f'Found {unique_count} unique frames out of {count} total frames.')

        # Detect objects in unique frames using YOLOv5
        # Change current working directory to where detect.py is located
        os.chdir('C:/pythonProject1/yolov7')

        # Run detect.py command
        os.system(f'python detect.py --source "C:/pythonProject1/unique_frames" --weights "C:/pythonProject1/yolov7/weights/best.pt" --conf 0.7 --save-txt')

        # Count the number of vehicle detections
        detections_dict = count_vehicle_detections(video_path, '/path/to/yolov7/runs/detect/exp/labels')
        half_dict = {k: int(v) for k, v in detections_dict.items()}
        bike = int(half_dict["Bike"])
        car = int(half_dict["Car"])
        ht = int(half_dict["Heavy truck"])
        totals = bike + car + ht

        def maintance(total):
            if total <= 50:
                st.write("Total number of vehicle traveled is", totals, "no maintenance is required")
            else:
                st.write("Total number of vehicle traveled is", totals, " maintenance is required")

        # Display the number of vehicle detections
        st.write(f'Found {half_dict} unique frames out of {count} total frames.')
        maintance(totals)


if __name__ == "__main__":
    main()