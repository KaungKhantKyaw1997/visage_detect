from flask import Flask, request, jsonify, send_file, url_for
import cv2
import os
from deepface import DeepFace
import tempfile
from imutils.object_detection import non_max_suppression
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) and file_path.endswith('.jpg'):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

@app.route('/detect_and_recognize_faces', methods=['POST'])
def detect_and_recognize_faces():
    cleanup_temp_files()

    data = request.json
    group_photos_dir = data.get('group_photos_dir')
    known_faces_dir = data.get('known_faces_dir')

    if not group_photos_dir or not os.path.exists(group_photos_dir):
        return jsonify(error=f"Group photos directory '{group_photos_dir}' does not exist"), 400
    if not known_faces_dir or not os.path.exists(known_faces_dir):
        return jsonify(error=f"Known faces directory '{known_faces_dir}' does not exist"), 400

    processed_photo_urls = []
    for group_photo_filename in os.listdir(group_photos_dir):
        group_photo_path = os.path.join(group_photos_dir, group_photo_filename)
        if not any(group_photo_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            continue

        try:
            processed_photo, output_path = process_group_photo(group_photo_path, known_faces_dir)
            processed_photo_url = url_for('serve_processed_image', filename=output_path, _external=True)
            processed_photo_urls.append(processed_photo_url)
        except Exception as e:
            print(f"Error processing {group_photo_path}: {e}")

    return jsonify(processed_photo_urls)

def process_group_photo(group_photo_path, known_faces_dir):
    group_photo = cv2.imread(group_photo_path)
    if group_photo is None:
        raise ValueError(f"Failed to read image: {group_photo_path}")

    detected_faces = DeepFace.extract_faces(img_path=group_photo_path, enforce_detection=False)

    if not detected_faces:
        raise ValueError("No faces detected.")

    processed_photo = group_photo.copy()
    known_faces = load_known_faces(known_faces_dir)

    rectangles = []
    for face in detected_faces:
        if 'facial_area' in face:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            rectangles.append((x, y, x + w, y + h))

    rectangles = sorted(rectangles, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]), reverse=True)
    rects_np = np.array(rectangles)

    pick = non_max_suppression(rects_np, probs=None, overlapThresh=0.3)

    labeled_faces = set()
    for (xA, yA, xB, yB) in pick:
        face_img = group_photo[yA:yB, xA:xB]
        best_match = recognize_face(face_img, known_faces)

        if best_match and best_match not in labeled_faces:
            labeled_faces.add(best_match)
            cv2.rectangle(processed_photo, (xA, yA), (xB, yB), (0, 255, 0), 2)
            label_position = (xA, yA - 10 if yA - 10 > 10 else yA + 10)
            cv2.putText(processed_photo, best_match, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    output_path = os.path.basename(temp_output_file.name)
    cv2.imwrite(temp_output_file.name, processed_photo)

    return processed_photo, output_path

def load_known_faces(known_faces_dir):
    known_faces = {}
    for filename in os.listdir(known_faces_dir):
        if any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            person_name = os.path.splitext(filename)[0]
            known_faces[person_name] = os.path.join(known_faces_dir, filename)
    return known_faces

def recognize_face(face_img, known_faces):
    best_match = "Unknown"
    best_score = float('inf')

    for person_name, person_image_path in known_faces.items():
        try:
            result = DeepFace.verify(face_img, person_image_path, enforce_detection=False, threshold=15)
            score = result['distance']
            
            if result['verified'] and score < best_score:
                best_score = score
                best_match = person_name

        except Exception as e:
            print(f"Error verifying face with {person_name}: {e}")

    return best_match if best_match != "Unknown" else None

@app.route('/serve_processed_image/<filename>')
def serve_processed_image(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
