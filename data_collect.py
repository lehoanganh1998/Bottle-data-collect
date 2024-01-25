from ultralytics import YOLO
import cv2
import time
import os

model = YOLO('/home/lee/Documents/HyperSort/data_collect/best_weights/best_inference_yolov8x.pt')
raw_dir = "/home/lee/Documents/HyperSort/data_collect/save_files/raw"
pred_dir = "/home/lee/Documents/HyperSort/data_collect/save_files/pred"
duration = 2

font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font style for displaying text
font_scale = 0.5  # Adjust font size
font_color = (0, 0, 255)  # Blue text color

cap = cv2.VideoCapture(4)
last_try_time = time.time()

while True:
    success, frame = cap.read()

    if success:
        results = model(frame, show=True, conf=0.6, iou=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()

        if time.time() - last_try_time >= duration:
            if not boxes:
                print("No bottle detected")
            else:
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
                    
                # Save the final frame with all boxes drawn
                print("Number of bottle: {count}")
                cv2.imwrite(pred, frame)

                # Create a single TXT file with all boxes' info
                txt_pred = pred.replace(".jpg", ".txt")
                with open(txt_pred, "w") as f:
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        confidence = conf
                        f.write(f"{x1},{y1},{x2},{y2},{confidence}\n")


            last_try_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
