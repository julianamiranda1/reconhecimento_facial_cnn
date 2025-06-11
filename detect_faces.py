# src/detect_faces.py
import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_and_crop_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir): continue

        output_person_dir = os.path.join(output_dir, person)
        os.makedirs(output_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)

            if results.detections:
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    bw, bh = int(bbox.width * w), int(bbox.height * h)
                    face = img[y:y+bh, x:x+bw]

                    output_path = os.path.join(output_person_dir, f"{os.path.splitext(img_name)[0]}_face{i}.jpg")
                    cv2.imwrite(output_path, face)

if __name__ == "__main__":
    detect_and_crop_faces("rostos_participantes", "rostos_cortados")
