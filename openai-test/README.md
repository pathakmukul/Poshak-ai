# OpenAI Test Images

This folder contains test images for the OpenAI outfit generation feature.

## Folder Structure

- `MODEL/` - Place model/person images here
- `SHIRT/` - Place shirt images here
- `PANT/` - Place pants images here
- `SHOES/` - Place shoes images here
- `Accessories/` - Place accessories images here

## Adding Images

1. Add images to the appropriate folder based on category
2. Supported formats: JPG, JPEG, PNG, WEBP
3. Images will automatically appear in the dropdown menus in the OpenAI Switch interface

## Usage

1. Start the local OpenAI backend:
   ```bash
   cd local-backend
   ./run-openai-backend.sh
   ```

2. Make sure to set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. Access the OpenAI Switch feature from the main app page