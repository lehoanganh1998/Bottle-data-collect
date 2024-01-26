from ultralytics import YOLO
import cv2
import time
import os

# Setup
model = YOLO('./best_weights/best_inference_yolov8x.pt')
raw_dir = "./save_files/raw"
pred_dir = "./save_files/pred"

# Time between captured frames
duration = 30

# Camera index
cam_index = 8

# Text setup
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font style for displaying text
font_scale = 0.5  # Adjust font size
font_color = (0, 0, 255)  # Red text color

# initialize camera
cap = cv2.VideoCapture(cam_index)
last_try_time = time.time()

# Initialize model
run_model = True


while True:
    success, frame = cap.read()

    if success:
        if run_model:
            # results = model(frame, show=True, conf=0.6, iou=0.5, verbose=False)
            results = model(frame, show=False, conf=0.6, iou=0.5, verbose=False, device='cpu')
            boxes = results[0].boxes.xyxy.tolist()
            confidences = results[0].boxes.conf.tolist()

            # Save frame only when boxes are detected
            if boxes:
                count = 0
                raw = os.path.join(raw_dir, f"raw_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                cv2.imwrite(raw, frame)
                for box, conf in zip(boxes, confidences):
                    count+=1
                    x1, y1, x2, y2 = box
                    confidence = conf
                    pred = os.path.join(pred_dir, f"pred_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Calculate text position above the box
                    text_x, text_y = int((x1 + x2) / 2), int(y1 - 10)  # Adjust these offsets to position the text

                    # Display confidence score as text
                    text = f"{confidence:.2f}"  # Format confidence to two decimal places
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, 2)
                    
                # Number of bottle counter
                print(f"Number of bottle: {count}")
                
                # Save the final frame with all boxes drawn
                cv2.imwrite(pred, frame)

                # Create a single TXT file with all boxes' info
                txt_pred = os.path.join(raw_dir, f"pred_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
                with open(txt_pred, "w") as f:
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        confidence = conf
                        f.write(f"{x1},{y1},{x2},{y2},{confidence}\n")
            else:
                print('No bottle detected')
                
            # Turn off model to prevent overheat
            run_model = False
            
        # Check if it's time to run the model again
        if time.time() - last_try_time >= duration:
            last_try_time = time.time()
            run_model = True  # Set the flag to True to enable model execution in the next loop
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
