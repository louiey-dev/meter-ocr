# MeterVision OCR Project Context

## Project Overview
MeterVision OCR is a specialized vision system designed to automate the reading of utility meters (Water, Gas, Electric). It uses a two-stage approach:
1. **Detection:** Identifying the Region of Interest (ROI) containing the digits.
2. **OCR:** Extracting the digit string from the identified ROI.

## Tech Stack
- **Frontend:** React 19, Vite, Tailwind CSS 4.
- **AI Engine:** Gemini 3 Flash (Multimodal Vision).
- **Animations:** Motion (formerly Framer Motion).
- **Icons:** Lucide React.

## Current Implementation Details
- **Multimodal Pipeline:** The app sends the full meter image to Gemini with a specialized prompt that returns digits, confidence, meter type, and normalized ROI coordinates `[ymin, xmin, ymax, xmax]`.
- **Image Cropping:** Uses a client-side `<canvas>` to crop the original high-resolution image based on the AI-provided ROI.
- **Record Saving:** The "Save Record" feature downloads the cropped digit area as a `.jpg` file named `meter_[digits]_crop.jpg`.
- **Modeling Lab:** A dedicated training environment using TensorFlow.js for on-device model development.
    - Supports bulk image upload and manual labeling.
    - Trains a lightweight CNN architecture optimized for ESP32-S3 constraints.
    - Real-time training metrics (Loss/Accuracy) visualization using Recharts.
    - Exports models in TF.js format (convertible to TFLite/ESP-DL).

## Design Principles
- **Hardware Aesthetic:** Dark, technical interface with high-contrast accents (#FF4444).
- **Precision Focus:** Uses monospace fonts (JetBrains Mono) for data and technical logs.
- **Real-time Feedback:** Includes a scanning animation and live system logs to communicate the vision engine's state.

## Critical Rules & Fixes
- **Defensive Coding:** Always use optional chaining on AI response objects (e.g., `result.confidence?.toLowerCase()`) to prevent crashes from partial JSON responses.
- **ROI Validation:** Ensure `result.roi` is a valid array of 4 numbers before attempting canvas operations.
- **Image Security:** Always use `referrerPolicy="no-referrer"` on `<img>` tags.

## Future Roadmap (Based on User Design)
- **Phase 2:** Transition to custom detection models (SSD/EfficientDet) if higher local performance is needed.
- **Phase 3:** Explore CRNN for specialized analog meter digit variations.
- **Phase 4:** perspective correction and contrast enhancement preprocessing.
