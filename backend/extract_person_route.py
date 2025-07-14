@app.route('/extract-person', methods=['POST'])
def extract_person():
    """Extract person from image using MediaPipe and return visualization"""
    try:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 500
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract person
        if PERSON_EXTRACTION_CONFIG.get('use_person_extraction', True):
            cropped_person, person_bbox, person_mask = extract_person_from_image(
                image_rgb, 
                padding_percent=PERSON_EXTRACTION_CONFIG.get('padding_percent', 10)
            )
            
            # Create visualization
            viz_img = image_rgb.copy()
            
            if cropped_person is not None:
                # Draw bounding box
                x, y, w, h = person_bbox
                cv2.rectangle(viz_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Apply mask overlay
                if person_mask is not None:
                    mask_overlay = np.zeros_like(viz_img)
                    mask_overlay[person_mask > 0] = [0, 255, 0]
                    viz_img = cv2.addWeighted(viz_img, 0.7, mask_overlay, 0.3, 0)
                
                # Add text
                cv2.putText(viz_img, "Person Detected - Processing this region", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(viz_img, f"Bbox: {person_bbox}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return jsonify({
                    'person_detected': True,
                    'person_viz': image_to_base64(viz_img),
                    'bbox': person_bbox
                })
            else:
                cv2.putText(viz_img, "No Person Detected - Will process full image", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                return jsonify({
                    'person_detected': False,
                    'person_viz': image_to_base64(viz_img)
                })
        else:
            return jsonify({
                'person_detected': False,
                'message': 'Person extraction disabled'
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500