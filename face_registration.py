import cv2
import os
import time


def face_register(user_name,single_pic):
    # Create a folder to save the captured images
    # user_name = input("Enter candidate name:")
    if single_pic == False:
        output_folder = f'Dataset/{user_name}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    else:
        output_folder = ""

    # Load the pre-trained deep learning face detection model
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


    # Access the computer's camera
    cap = cv2.VideoCapture(0)

    count = 0
    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Create a blob from the frame and perform a forward pass to detect faces
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        # Iterate over the detected faces and save them
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold to filter weak detections
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box coordinates are within the frame dimensions
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                face = frame[startY:endY, startX:endX]
            
                elapsed_time = time.time() - start_time
                print(elapsed_time)
                flag=0
                if single_pic == True and elapsed_time >= 5 :
                    cv2.imwrite('test_face.jpg', face)
                    flag=1
                    break

                if count <= 20 and elapsed_time >= 1 and single_pic == False:
                    count += 1
                    start_time = time.time()
                    face_filename = os.path.join(output_folder, f'face_{count}.jpg')
                    cv2.imwrite(face_filename, face)
                    print(f'Saved {face_filename}')
                
        if flag==1:
            break
        if count >= 20:
            break

        # Display the frame with detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)


        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
