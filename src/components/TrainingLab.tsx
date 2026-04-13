import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Upload, Play, Save, Trash2, BarChart3, Settings, Database, Cpu, Download, Info, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import JSZip from 'jszip';

interface TrainingData {
  id: string;
  image: string;
  label: string;
  roi?: [number, number, number, number]; // [ymin, xmin, ymax, xmax] normalized 0-1000
}

// IndexedDB Helper
const DB_NAME = 'MeterVisionDB';
const STORE_NAME = 'dataset';

const initDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const saveToDB = async (data: TrainingData[]) => {
  const db = await initDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);
  
  // Clear and re-save
  await new Promise<void>((resolve) => {
    const clearReq = store.clear();
    clearReq.onsuccess = () => resolve();
  });

  for (const item of data) {
    store.add(item);
  }
};

const loadFromDB = async (): Promise<TrainingData[]> => {
  const db = await initDB();
  const tx = db.transaction(STORE_NAME, 'readonly');
  const store = tx.objectStore(STORE_NAME);
  return new Promise((resolve) => {
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
  });
};

export default function TrainingLab({ onModelTrained }: { onModelTrained?: (model: tf.LayersModel) => void }) {
  const [dataset, setDataset] = useState<TrainingData[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<{ epoch: number; loss: number; acc: number }[]>([]);
  const [epochs, setEpochs] = useState(20);
  const [batchSize, setBatchSize] = useState(8);
  const [showExportInfo, setShowExportInfo] = useState(false);
  const [trainedModel, setTrainedModel] = useState<tf.LayersModel | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [editingRoiId, setEditingRoiId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const roiCanvasRef = useRef<HTMLCanvasElement>(null);
  const [roiStart, setRoiStart] = useState<{x: number, y: number} | null>(null);
  const [currentRoi, setCurrentRoi] = useState<[number, number, number, number] | null>(null);

  // Load dataset on mount
  useEffect(() => {
    loadFromDB().then(data => {
      if (data && data.length > 0) {
        setDataset(data);
      }
      setIsLoaded(true);
    });
  }, []);

  // Auto-save dataset changes
  useEffect(() => {
    if (isLoaded) {
      setIsSaving(true);
      const timer = setTimeout(() => {
        saveToDB(dataset).finally(() => setIsSaving(false));
      }, 1000); // Debounce save
      return () => clearTimeout(timer);
    }
  }, [dataset, isLoaded]);

  const handleBulkUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList) return;
    
    const files = Array.from(fileList) as File[];
    const newItems: TrainingData[] = [];

    for (const file of files) {
      const reader = new FileReader();
      const promise = new Promise<void>((resolve) => {
        reader.onload = async (event) => {
          newItems.push({
            id: Math.random().toString(36).substr(2, 9),
            image: event.target?.result as string,
            label: '',
          });
          resolve();
        };
      });
      reader.readAsDataURL(file);
      await promise;
    }
    setDataset(prev => [...prev, ...newItems]);
  };

  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const clearDataset = async () => {
    setDataset([]);
    const db = await initDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).clear();
    setShowClearConfirm(false);
  };

  const exportProject = () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `metervision_project_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  const importProject = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target?.result as string);
          if (Array.isArray(data)) {
            setDataset(data);
          }
        } catch (err) {
          console.error('Invalid project file', err);
        }
      };
      reader.readAsText(file);
    }
  };

  const stats = {
    total: dataset.length,
    labeled: dataset.filter(d => d.label.length > 0).length,
    avgLength: dataset.length > 0 
      ? (dataset.reduce((acc, curr) => acc + curr.label.length, 0) / dataset.length).toFixed(1)
      : 0
  };

  const trainModel = async () => {
    if (dataset.length === 0) return;
    
    setIsTraining(true);
    setLogs([]);
    setProgress(0);
    setTrainedModel(null);

    try {
      const imgSize = 96;
      const xs: tf.Tensor[] = [];
      const ys_digits: number[][] = Array.from({ length: 6 }, () => []);
      const ys_roi: number[][] = [];

      for (const item of dataset) {
        const img = new Image();
        img.src = item.image;
        await new Promise(r => img.onload = r);
        
        const h = img.height;
        const w = img.width;
        const maxDim = Math.max(h, w);
        const padTop = Math.floor((maxDim - h) / 2);
        const padLeft = Math.floor((maxDim - w) / 2);

        const baseTensor = tf.tidy(() => {
          const tensor = tf.browser.fromPixels(img);
          const padded = tensor.pad([[padTop, maxDim - h - padTop], [padLeft, maxDim - w - padLeft], [0, 0]]);
          return padded.resizeBilinear([imgSize, imgSize])
            .toFloat()
            .div(tf.scalar(255.0));
        });

        // Data Augmentation: 4 versions per image
        // 1. Original
        // 2. Random Shift + Brightness
        // 3. Random Noise + Contrast
        // 4. Slight Rotation
        const augmentations = [
          { tensor: baseTensor, shift: [0, 0] },
          { 
            tensor: tf.tidy(() => {
              const brightness = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
              return baseTensor.mul(tf.scalar(brightness)).clipByValue(0, 1);
            }),
            shift: [0, 0]
          },
          {
            tensor: tf.tidy(() => {
              const noise = tf.randomNormal([imgSize, imgSize, 3], 0, 0.03);
              return baseTensor.add(noise).clipByValue(0, 1);
            }),
            shift: [0, 0]
          },
          {
            tensor: tf.tidy(() => {
              const angle = (Math.random() - 0.5) * 0.15; // +/- ~4 degrees
              // Note: tf.image.rotateWithOffset is not always available in all backends, 
              // but we can use a simple trick or just skip if it fails.
              // For now, let's use a simple horizontal flip as a safe alternative if rotation is tricky,
              // but meters are usually oriented one way. Let's stick to brightness/noise if rotation is complex.
              // Actually, let's try a random contrast adjustment instead.
              const contrast = 0.8 + Math.random() * 0.4;
              const mean = baseTensor.mean();
              return baseTensor.sub(mean).mul(tf.scalar(contrast)).add(mean).clipByValue(0, 1);
            }),
            shift: [0, 0]
          }
        ];

        for (const aug of augmentations) {
          xs.push(aug.tensor.expandDims());
          
          const labelStr = item.label.padEnd(6, 'A').replace(/A/g, '10');
          for (let i = 0; i < 6; i++) {
            const digit = parseInt(labelStr[i]);
            ys_digits[i].push(isNaN(digit) ? 10 : digit);
          }

          const roi = item.roi || [400, 100, 600, 900];
          // Adjust ROI to padded square coordinates
          const [ymin, xmin, ymax, xmax] = roi;
          const ymin_pad = ((ymin / 1000) * h + padTop) / maxDim;
          const xmin_pad = ((xmin / 1000) * w + padLeft) / maxDim;
          const ymax_pad = ((ymax / 1000) * h + padTop) / maxDim;
          const xmax_pad = ((xmax / 1000) * w + padLeft) / maxDim;
          
          ys_roi.push([ymin_pad, xmin_pad, ymax_pad, xmax_pad]);
        }
        
        baseTensor.dispose();
      }

      const trainX = tf.concat(xs);
      const trainYsDigits = ys_digits.map(y => tf.oneHot(tf.tensor1d(y, 'int32'), 11));
      const trainYRoi = tf.tensor2d(ys_roi);

      // Advanced Architecture with Batch Normalization
      const input = tf.input({ shape: [imgSize, imgSize, 3] });
      
      let x = tf.layers.conv2d({ kernelSize: 3, filters: 32, padding: 'same' }).apply(input) as tf.SymbolicTensor;
      x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
      x = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(x) as tf.SymbolicTensor;
      
      x = tf.layers.conv2d({ kernelSize: 3, filters: 64, padding: 'same' }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
      x = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(x) as tf.SymbolicTensor;

      x = tf.layers.conv2d({ kernelSize: 3, filters: 128, padding: 'same' }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
      x = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(x) as tf.SymbolicTensor;
      
      x = tf.layers.flatten().apply(x) as tf.SymbolicTensor;
      x = tf.layers.dense({ units: 512 }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.leakyReLU({ alpha: 0.1 }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.dropout({ rate: 0.3 }).apply(x) as tf.SymbolicTensor;

      const digitOutputs = Array.from({ length: 6 }, (_, i) => 
        tf.layers.dense({ units: 11, activation: 'softmax', name: `digit_${i}` }).apply(x) as tf.SymbolicTensor
      );

      const roiOutput = tf.layers.dense({ units: 4, activation: 'sigmoid', name: 'roi' }).apply(x) as tf.SymbolicTensor;

      const model = tf.model({ inputs: input, outputs: [...digitOutputs, roiOutput] });

      model.compile({
        optimizer: tf.train.adam(0.0002),
        loss: {
          ...Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`digit_${i}`, 'categoricalCrossentropy'])),
          roi: 'meanSquaredError'
        },
        lossWeights: {
          ...Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`digit_${i}`, 1.0])),
          roi: 5.0 // Give ROI more weight to fix the "not fit" issue
        },
        metrics: ['accuracy'],
      } as any);

      const targetYs: { [key: string]: tf.Tensor } = { roi: trainYRoi };
      trainYsDigits.forEach((y, i) => { targetYs[`digit_${i}`] = y; });

      await model.fit(trainX, targetYs, {
        epochs,
        batchSize,
        callbacks: {
          onEpochEnd: (epoch, log) => {
            // Average loss across all heads
            const avgLoss = (
              (log?.digit_0_loss || 0) + 
              (log?.digit_1_loss || 0) + 
              (log?.digit_2_loss || 0) + 
              (log?.digit_3_loss || 0) + 
              (log?.digit_4_loss || 0) +
              (log?.digit_5_loss || 0) +
              (log?.roi_loss || 0)
            ) / 7;
            
            // Log ROI loss specifically for debugging
            if (epoch % 5 === 0) {
              console.log(`Epoch ${epoch}: ROI Loss = ${log?.roi_loss?.toFixed(6)}`);
            }

            setLogs(prev => [...prev, { epoch, loss: avgLoss, acc: log?.acc || 0 }]);
            setProgress(((epoch + 1) / epochs) * 100);
          }
        }
      });

      setTrainedModel(model);
      if (onModelTrained) onModelTrained(model);
      setShowExportInfo(true);
      
      trainX.dispose();
      Object.values(targetYs).forEach(t => t.dispose());
    } catch (err) {
      console.error("Training Error:", err);
    } finally {
      setIsTraining(false);
    }
  };

  const exportBundle = async () => {
    if (!trainedModel) return;

    const zip = new JSZip();
    
    // 1. Save TF.js Model to memory
    const modelSaveResult = await trainedModel.save(tf.io.withSaveHandler(async (artifacts) => {
      // Add model.json
      zip.file('model.json', JSON.stringify(artifacts.modelTopology));
      
      // Add weights
      if (artifacts.weightData) {
        const weightData = artifacts.weightData instanceof ArrayBuffer 
          ? artifacts.weightData 
          : new Uint8Array(artifacts.weightData.reduce((acc, curr) => acc + curr.byteLength, 0));
        
        if (!(artifacts.weightData instanceof ArrayBuffer)) {
          let offset = 0;
          const uint8 = weightData as Uint8Array;
          for (const buffer of artifacts.weightData) {
            uint8.set(new Uint8Array(buffer), offset);
            offset += buffer.byteLength;
          }
        }
        
        zip.file('weights.bin', weightData);
      }
      
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));

    // 2. Add TFLite Conversion Script (Python)
    const pythonScript = `
import os
import sys
import subprocess

# 1. CRITICAL: Set this BEFORE any imports to bypass Protobuf version checks
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
except ImportError:
    print("Dependencies missing. Installing Surgical Combo...")
    # Install in specific order to avoid ResolutionImpossible
    install_package("protobuf==4.25.3")
    install_package("tensorflow==2.16.1")
    install_package("tensorflow-decision-forests==1.9.0")
    install_package("tensorflowjs==4.20.0")
    import tensorflow as tf
    import tensorflowjs as tfjs

print("\\n--- MeterVision OCR Converter ---")
print("Loading TF.js model from model.json...")

try:
    # 1. Load the TF.js model
    # We use a try-except block to catch the Protobuf VersionError specifically
    model = tfjs.converters.load_keras_model('model.json')
    
    print("Model loaded successfully.")

    # 2. Convert to TFLite
    print("Converting to TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization for ESP32-S3 (Optional: can add quantization here)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # 3. Save the .tflite file
    output_path = 'model.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print("\\n" + "="*50)
    print(f"SUCCESS: {output_path} generated!")
    print("Next step: Use ESP-DL tools to convert this to C++ arrays.")
    print("="*50)

except Exception as e:
    print("\\n" + "!"*50)
    print("CONVERSION ERROR DETECTED")
    print("!"*50)
    print(f"Details: {e}")
    
    if "Protobuf" in str(e) or "VersionError" in str(e):
        print("\\nREMEDY: Your Python environment has a Protobuf version conflict.")
        print("Please run the following command and then try again:")
        print("\\n    pip install --upgrade protobuf")
        print("\\nAlternatively, try creating a fresh virtual environment.")
    else:
        print("\\nIf you continue to have issues, ensure you are using Python 3.10 or 3.11.")
    `.trim();
    
    zip.file('convert_to_tflite.py', pythonScript);

    // 3. Add requirements.txt
    const requirements = `
# Legacy Stable Combo (Best for Python 3.12 + Linux)
# This avoids the Protobuf 6.x conflict entirely
numpy<2.0.0
tensorflow==2.15.0
tensorflow-decision-forests==1.8.1
tensorflowjs==4.17.0
protobuf==4.25.3
    `.trim();
    zip.file('requirements.txt', requirements);

    // 4. Add README
    const readme = `
# Meter OCR Model Bundle (ESP32-S3)

This bundle contains your trained model and the tools to convert it for ESP32-S3.

## ⚠️ CRITICAL: Environment Setup
Do NOT run this inside an ESP-IDF or ESP-WHO directory. 

1. Create a NEW folder outside your ESP projects:
   mkdir ~/meter_model_convert
   cd ~/meter_model_convert

2. Create a FRESH virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install the "Legacy Stable" stack (Copy/Paste these):
   pip install "numpy<2.0.0"
   pip install "protobuf==4.25.3"
   pip install "tensorflow==2.15.0"
   pip install "tensorflow-decision-forests==1.8.1"
   pip install "tensorflowjs==4.17.0"

4. Run the conversion script:
   python convert_to_tflite.py

## Troubleshooting
If you get "Temporary failure in name resolution":
- Check your internet connection and DNS settings.

If you get "ResolutionImpossible":
- Ensure you are using a FRESH virtual environment.
- Run the commands in Step 3 one by one.

## Next Steps
Use the resulting model.tflite with Espressif's ESP-DL tools.
    `.trim();
    
    zip.file('README.txt', readme);

    // 4. Download Zip
    const content = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(content);
    link.download = 'meter_ocr_esp32_bundle.zip';
    link.click();
  };

  const updateLabel = (id: string, label: string) => {
    const val = label.replace(/[^0-9]/g, '').slice(0, 6);
    setDataset(prev => prev.map(item => item.id === id ? { ...item, label: val } : item));
  };

  const removeItem = (id: string) => {
    setDataset(prev => prev.filter(item => item.id !== id));
  };

  const startRoiLabeling = (id: string) => {
    setEditingRoiId(id);
    const item = dataset.find(d => d.id === id);
    setCurrentRoi(item?.roi || null);
    setRoiStart(null);
  };

  const saveRoi = () => {
    if (editingRoiId && currentRoi) {
      setDataset(prev => prev.map(item => item.id === editingRoiId ? { ...item, roi: currentRoi } : item));
    }
    setEditingRoiId(null);
  };

  const handleRoiMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = roiCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 1000;
    const y = ((e.clientY - rect.top) / rect.height) * 1000;
    setRoiStart({ x, y });
    setCurrentRoi([y, x, y, x]);
  };

  const handleRoiMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!roiStart) return;
    const canvas = roiCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 1000;
    const y = ((e.clientY - rect.top) / rect.height) * 1000;
    
    setCurrentRoi([
      Math.min(roiStart.y, y),
      Math.min(roiStart.x, x),
      Math.max(roiStart.y, y),
      Math.max(roiStart.x, x)
    ]);
  };

  const handleRoiMouseUp = () => {
    setRoiStart(null);
  };

  useEffect(() => {
    if (editingRoiId && roiCanvasRef.current) {
      const canvas = roiCanvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const item = dataset.find(d => d.id === editingRoiId);
      if (!item) return;

      const img = new Image();
      img.src = item.image;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        if (currentRoi) {
          ctx.strokeStyle = '#FF4444';
          ctx.lineWidth = 4;
          const [ymin, xmin, ymax, xmax] = currentRoi;
          ctx.strokeRect(
            (xmin / 1000) * img.width,
            (ymin / 1000) * img.height,
            ((xmax - xmin) / 1000) * img.width,
            ((ymax - ymin) / 1000) * img.height
          );
          ctx.fillStyle = 'rgba(255, 68, 68, 0.2)';
          ctx.fillRect(
            (xmin / 1000) * img.width,
            (ymin / 1000) * img.height,
            ((xmax - xmin) / 1000) * img.width,
            ((ymax - ymin) / 1000) * img.height
          );
        }
      };
    }
  }, [editingRoiId, currentRoi, dataset]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e0e0e0] font-mono p-4 md:p-8">
      <header className="max-w-7xl mx-auto mb-8 flex items-center justify-between border-b border-[#333] pb-6">
        <div>
          <div className="flex items-center gap-2 text-[#FF4444] mb-2">
            <Cpu className="w-4 h-4" />
            <span className="text-[10px] tracking-[0.2em] uppercase font-bold">Training Studio // ESP32-S3 Target</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tighter text-white uppercase">Modeling <span className="text-[#FF4444]">Lab</span></h1>
        </div>
        <div className="flex items-center gap-4">
          {isSaving && (
            <div className="flex items-center gap-2 text-[10px] text-[#444] uppercase animate-pulse">
              <RefreshCw className="w-3 h-3" />
              Auto-saving...
            </div>
          )}
          <div className="flex gap-2">
            <button 
              onClick={exportProject}
              disabled={dataset.length === 0}
              className="flex items-center gap-2 px-3 py-2 bg-[#1a1b1e] border border-[#333] rounded-lg text-[10px] uppercase tracking-widest hover:bg-[#222] transition-colors disabled:opacity-50"
              title="Export project as JSON"
            >
              <Download className="w-3 h-3" />
              Export
            </button>
            <label className="flex items-center gap-2 px-3 py-2 bg-[#1a1b1e] border border-[#333] rounded-lg text-[10px] uppercase tracking-widest hover:bg-[#222] transition-colors cursor-pointer">
              <Upload className="w-3 h-3" />
              Import
              <input type="file" onChange={importProject} className="hidden" accept=".json" />
            </label>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-4 py-2 bg-[#FF4444] text-white rounded-lg text-xs uppercase tracking-widest hover:bg-[#FF6666] transition-colors"
            >
              <Upload className="w-4 h-4" />
              Import Images
            </button>
          </div>
          <input type="file" ref={fileInputRef} multiple onChange={handleBulkUpload} className="hidden" accept="image/*" />
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Dataset Management */}
        <div className="lg:col-span-8 space-y-6">
          <div className="bg-[#151619] rounded-xl border border-[#333] p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xs uppercase tracking-[0.3em] text-[#666] font-bold flex items-center gap-2">
                <Database className="w-4 h-4" />
                Dataset ({dataset.length} Images)
              </h2>
              {dataset.length > 0 && (
                <div className="flex items-center gap-2">
                  {showClearConfirm ? (
                    <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 px-2 py-1 rounded">
                      <span className="text-[8px] text-red-500 uppercase font-bold">Confirm?</span>
                      <button 
                        onClick={clearDataset}
                        className="text-[8px] uppercase tracking-widest text-white bg-red-500 px-2 py-0.5 rounded hover:bg-red-600 transition-colors"
                      >
                        Yes
                      </button>
                      <button 
                        onClick={() => setShowClearConfirm(false)}
                        className="text-[8px] uppercase tracking-widest text-[#666] hover:text-white transition-colors"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button 
                      onClick={() => setShowClearConfirm(true)}
                      className="text-[10px] uppercase tracking-widest text-red-500 hover:text-red-400 flex items-center gap-1 transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                      Clear All
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Training Tips */}
            <div className="mb-6 p-4 bg-[#FF4444]/5 border border-[#FF4444]/20 rounded-lg">
              <div className="flex items-center gap-2 text-[#FF4444] mb-2">
                <Info className="w-5 h-5" />
                <span className="text-[18px] uppercase font-bold tracking-widest">Training Tips for ESP32-S3</span>
              </div>
      <ul className="text-[18px] text-[#888] space-y-2 list-disc list-inside">
        <li><span className="text-[#FF4444]">ROI Accuracy:</span> Draw the ROI box tightly around the digits. If the box is too loose, the model will struggle.</li>
        <li><span className="text-[#FF4444]">Data Quantity:</span> For 6-digit detection, aim for at least <span className="text-white">50-100 images</span> with varied lighting.</li>
        <li><span className="text-[#FF4444]">Alignment:</span> If digits are "shifting" (e.g. 00123 to 01230), ensure your ROI boxes are consistently centered.</li>
      </ul>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
              {dataset.map((item) => (
                <div key={item.id} className="bg-black rounded-lg border border-[#222] overflow-hidden group relative">
                  <img src={item.image} className="w-full aspect-square object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                  <div className="p-2 space-y-2">
                    <input 
                      type="text" 
                      placeholder="000000"
                      maxLength={6}
                      value={item.label}
                      onChange={(e) => updateLabel(item.id, e.target.value)}
                      className={`w-full bg-[#111] border ${item.label.length >= 5 ? 'border-green-500/30' : 'border-[#333]'} text-[10px] p-1 rounded focus:border-[#FF4444] outline-none text-center font-bold`}
                    />
                    <button 
                      onClick={() => startRoiLabeling(item.id)}
                      className={`w-full text-[8px] py-1 rounded border ${item.roi ? 'border-green-500/50 text-green-500' : 'border-[#333] text-[#666]'} hover:border-[#FF4444] hover:text-[#FF4444] transition-colors uppercase tracking-widest`}
                    >
                      {item.roi ? 'ROI Set' : 'Set ROI'}
                    </button>
                  </div>
                  <button 
                    onClick={() => removeItem(item.id)}
                    className="absolute top-1 right-1 p-1 bg-red-500/80 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="w-3 h-3 text-white" />
                  </button>
                </div>
              ))}
              {dataset.length === 0 && (
                <div className="col-span-full py-20 text-center border-2 border-dashed border-[#222] rounded-xl">
                  <p className="text-[#444] text-xs uppercase tracking-widest">No training data loaded</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Training Controls & Analytics */}
        <div className="lg:col-span-4 space-y-6">
          {/* Config */}
          <div className="bg-[#151619] rounded-xl border border-[#333] p-6">
            <h2 className="text-xs uppercase tracking-[0.3em] text-[#666] font-bold mb-6 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Hyperparameters
            </h2>
            <div className="space-y-4">
              <div>
                <label className="text-[10px] text-[#444] uppercase block mb-2">Epochs: {epochs}</label>
                <input 
                  type="range" min="5" max="100" value={epochs} 
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full accent-[#FF4444]"
                />
              </div>
              <div>
                <label className="text-[10px] text-[#444] uppercase block mb-2">Batch Size: {batchSize}</label>
                <input 
                  type="range" min="1" max="32" value={batchSize} 
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full accent-[#FF4444]"
                />
              </div>
              <button 
                disabled={isTraining || dataset.length === 0}
                onClick={trainModel}
                className={`w-full py-4 rounded-xl font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all ${
                  isTraining || dataset.length === 0
                  ? 'bg-[#222] text-[#444] cursor-not-allowed'
                  : 'bg-[#FF4444] text-white hover:bg-[#FF6666]'
                }`}
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Training... {Math.round(progress)}%
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start Training
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Metrics & Export */}
          <div className="bg-[#151619] rounded-xl border border-[#333] p-6 flex-1">
            <h2 className="text-xs uppercase tracking-[0.3em] text-[#666] font-bold mb-6 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Training Metrics
            </h2>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={logs}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                  <XAxis dataKey="epoch" stroke="#444" fontSize={10} />
                  <YAxis stroke="#444" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '10px' }}
                    itemStyle={{ color: '#FF4444' }}
                  />
                  <Line type="monotone" dataKey="loss" stroke="#FF4444" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="acc" stroke="#44FF44" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <AnimatePresence>
              {trainedModel && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-6 pt-6 border-t border-[#222] space-y-4"
                >
                  <div className="bg-green-500/10 border border-green-500/20 p-4 rounded-lg flex items-start gap-3">
                    <Info className="w-4 h-4 text-green-500 mt-0.5" />
                    <div>
                      <p className="text-[10px] text-green-500 font-bold uppercase tracking-widest mb-1">Training Complete</p>
                      <p className="text-[10px] text-[#888]">Model is ready for ESP32-S3 deployment.</p>
                    </div>
                  </div>
                  
                  <button 
                    onClick={exportBundle}
                    className="w-full py-3 bg-white text-black rounded-lg text-xs uppercase tracking-widest font-bold flex items-center justify-center gap-2 hover:bg-[#ccc] transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download Model Bundle
                  </button>

                  <div className="bg-black/50 p-4 rounded-lg border border-[#222]">
                    <div className="flex items-center gap-2 text-[10px] text-[#666] uppercase mb-2">
                      <Terminal className="w-3 h-3" />
                      Next Steps
                    </div>
                    <ol className="text-[9px] text-[#444] space-y-2 list-decimal ml-4">
                      <li>Unzip the bundle</li>
                      <li>Run <code className="text-[#FF4444]">pip install tensorflowjs</code></li>
                      <li>Execute <code className="text-[#FF4444]">python convert_to_tflite.py</code></li>
                      <li>Flash <code className="text-white font-bold">model.tflite</code> to ESP32-S3</li>
                    </ol>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      {/* ROI Editor Modal */}
      <AnimatePresence>
        {editingRoiId && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-[#151619] border border-[#333] rounded-xl overflow-hidden max-w-4xl w-full"
            >
              <div className="p-4 border-b border-[#333] flex justify-between items-center">
                <h3 className="text-sm font-bold uppercase tracking-widest">Draw ROI Box</h3>
                <div className="flex gap-2">
                  <button onClick={() => setEditingRoiId(null)} className="px-4 py-2 text-xs text-[#666] hover:text-white transition-colors">Cancel</button>
                  <button onClick={saveRoi} className="px-4 py-2 bg-[#FF4444] text-white text-xs rounded font-bold hover:bg-[#ff5555] transition-colors">Save ROI</button>
                </div>
              </div>
              <div className="p-4 flex flex-col items-center">
                <p className="text-[10px] text-[#666] mb-4 uppercase tracking-widest">Click and drag to select the digits area</p>
                <div className="relative border border-[#333] bg-black rounded overflow-hidden cursor-crosshair">
                  <canvas 
                    ref={roiCanvasRef}
                    onMouseDown={handleRoiMouseDown}
                    onMouseMove={handleRoiMouseMove}
                    onMouseUp={handleRoiMouseUp}
                    className="max-w-full max-h-[70vh] object-contain"
                  />
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

function RefreshCw(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
      <path d="M3 21v-5h5" />
    </svg>
  );
}