import React, { useState, useRef, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { GoogleGenAI } from "@google/genai";
import { Camera, Upload, RefreshCw, CheckCircle2, AlertCircle, Maximize2, FileText, Save, Cpu, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Initialize Gemini API
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

interface OCRResult {
  digits: string;
  confidence: string;
  meterType: string;
  roi?: [number, number, number, number]; // [ymin, xmin, ymax, xmax] normalized 0-1000
  engine?: 'gemini' | 'local';
}

interface BatchItem {
  id: string;
  image: string;
  name: string;
  result: OCRResult | null;
  error: string | null;
  isProcessing: boolean;
}

export default function MeterOCR({ trainedModel }: { trainedModel: tf.LayersModel | null }) {
  const [batch, setBatch] = useState<BatchItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [engine, setEngine] = useState<'gemini' | 'local'>('gemini');
  const [imgDisplayRect, setImgDisplayRect] = useState<{top: number, left: number, width: number, height: number} | null>(null);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  
  const currentItem = batch[currentIndex] || null;
  
  const imgRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const updateImgRect = () => {
    if (!imgRef.current) return;
    const img = imgRef.current;
    const container = img.parentElement;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;

    const imgAspect = imgWidth / imgHeight;
    const containerAspect = containerWidth / containerHeight;

    let displayWidth, displayHeight, displayLeft, displayTop;

    if (imgAspect > containerAspect) {
      displayWidth = containerWidth;
      displayHeight = containerWidth / imgAspect;
      displayLeft = 0;
      displayTop = (containerHeight - displayHeight) / 2;
    } else {
      displayHeight = containerHeight;
      displayWidth = containerHeight * imgAspect;
      displayTop = 0;
      displayLeft = (containerWidth - displayWidth) / 2;
    }

    setImgDisplayRect({
      top: (displayTop / containerHeight) * 100,
      left: (displayLeft / containerWidth) * 100,
      width: (displayWidth / containerWidth) * 100,
      height: (displayHeight / containerHeight) * 100
    });
  };

  useEffect(() => {
    window.addEventListener('resize', updateImgRect);
    return () => window.removeEventListener('resize', updateImgRect);
  }, []);

  // Auto-switch to local if model is provided and user wants to test
  useEffect(() => {
    if (trainedModel && engine === 'gemini') {
      // We don't auto-switch, but we could show a hint
    }
  }, [trainedModel]);

  const processItem = async (index: number) => {
    const item = batch[index];
    if (!item || item.isProcessing) return;

    setBatch(prev => prev.map((it, i) => i === index ? { ...it, isProcessing: true, error: null } : it));

    try {
      let result: OCRResult;
      if (engine === 'local') {
        if (!trainedModel) {
          throw new Error("No trained model found. Please train a model in the Modeling Lab first.");
        }
        result = await runLocalInference(item.image);
      } else {
        result = await runGeminiInference(item.image);
      }
      setBatch(prev => prev.map((it, i) => i === index ? { ...it, result, isProcessing: false } : it));
    } catch (err: any) {
      console.error("OCR Error:", err);
      setBatch(prev => prev.map((it, i) => i === index ? { ...it, error: err.message || "Failed to process image.", isProcessing: false } : it));
    }
  };

  const processBatch = async () => {
    if (isBatchProcessing) return;
    setIsBatchProcessing(true);
    
    for (let i = 0; i < batch.length; i++) {
      if (!batch[i].result) {
        setCurrentIndex(i);
        await processItem(i);
      }
    }
    
    setIsBatchProcessing(false);
  };

  const runLocalInference = async (imgData: string): Promise<OCRResult> => {
    const img = new Image();
    img.src = imgData;
    await new Promise(r => img.onload = r);

    const h = img.height;
    const w = img.width;
    const maxDim = Math.max(h, w);
    const padTop = Math.floor((maxDim - h) / 2);
    const padLeft = Math.floor((maxDim - w) / 2);

    const imgSize = 96;
    const tensor = tf.tidy(() => {
      const t = tf.browser.fromPixels(img);
      const padded = t.pad([[padTop, maxDim - h - padTop], [padLeft, maxDim - w - padLeft], [0, 0]]);
      return padded.resizeBilinear([imgSize, imgSize])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    });

    // Multi-head model returns an array of 7 tensors (6 digits + 1 ROI)
    const predictions = trainedModel!.predict(tensor) as tf.Tensor[];
    let finalDigits = "";
    let totalConfidence = 0;

    // First 6 are digits
    for (let i = 0; i < 6; i++) {
      const scores = await predictions[i].data();
      const maxScoreIndex = scores.indexOf(Math.max(...Array.from(scores)));
      
      if (maxScoreIndex < 10) {
        finalDigits += maxScoreIndex.toString();
        totalConfidence += scores[maxScoreIndex];
      }
    }

    // 7th is ROI [ymin, xmin, ymax, xmax] normalized 0-1 in padded square space
    const roiData = await predictions[6].data();
    
    // Map back to original image coordinates (0-1000)
    const mapBack = (val: number, pad: number, dim: number) => {
      const abs = val * maxDim - pad;
      return Math.max(0, Math.min(1000, Math.round((abs / dim) * 1000)));
    };

    const predictedRoi: [number, number, number, number] = [
      mapBack(roiData[0], padTop, h),
      mapBack(roiData[1], padLeft, w),
      mapBack(roiData[2], padTop, h),
      mapBack(roiData[3], padLeft, w)
    ];

    const avgConfidence = totalConfidence / (finalDigits.length || 1);

    const res: OCRResult = {
      digits: finalDigits || "---",
      confidence: avgConfidence > 0.8 ? 'High' : avgConfidence > 0.5 ? 'Medium' : 'Low',
      meterType: 'Local Detection + OCR',
      engine: 'local',
      roi: predictedRoi
    };

    tensor.dispose();
    predictions.forEach(p => p.dispose());
    return res;
  };

  const runGeminiInference = async (imgData: string): Promise<OCRResult> => {
    const base64Data = imgData.split(',')[1];
    
    const prompt = `
      You are a specialized Meter OCR system. 
      1. Identify the area containing the digits (the register).
      2. Extract the digits accurately.
      3. Identify the meter type (e.g., Analog, Digital, Water, Electric, Gas).
      4. Provide a confidence level (High, Medium, Low).
      5. Provide the bounding box (ROI) of the digits area in normalized coordinates [ymin, xmin, ymax, xmax] (0-1000).
      
      Return the result in JSON format:
      {
        "digits": "string",
        "confidence": "string",
        "meterType": "string",
        "roi": [number, number, number, number]
      }
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: {
        parts: [
          { inlineData: { mimeType: "image/jpeg", data: base64Data } },
          { text: prompt }
        ]
      },
      config: {
        responseMimeType: "application/json"
      }
    });

    const data = JSON.parse(response.text || '{}');
    return { ...data, engine: 'gemini' };
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newItems: BatchItem[] = [];
    const readers = Array.from(files).map((file: File) => {
      return new Promise<void>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          newItems.push({
            id: Math.random().toString(36).substr(2, 9),
            image: reader.result as string,
            name: file.name,
            result: null,
            error: null,
            isProcessing: false
          });
          resolve();
        };
        reader.readAsDataURL(file);
      });
    });

    Promise.all(readers).then(() => {
      setBatch(prev => [...prev, ...newItems]);
      if (batch.length === 0) setCurrentIndex(0);
    });
  };

  const handleSave = useCallback(() => {
    if (!currentItem || !currentItem.result?.roi || !Array.isArray(currentItem.result.roi) || currentItem.result.roi.length < 4) {
      setBatch(prev => prev.map((it, i) => i === currentIndex ? { ...it, error: "Cannot save record: ROI data is missing or invalid." } : it));
      return;
    }

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const [ymin, xmin, ymax, xmax] = currentItem.result!.roi!;
      
      // Convert normalized to pixel coordinates
      const x = (xmin / 1000) * img.width;
      const y = (ymin / 1000) * img.height;
      const width = ((xmax - xmin) / 1000) * img.width;
      const height = ((ymax - ymin) / 1000) * img.height;

      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(img, x, y, width, height, 0, 0, width, height);
      
      // Trigger download
      const link = document.createElement('a');
      link.download = `meter_${currentItem.result!.digits}_crop.jpg`;
      link.href = canvas.toDataURL('image/jpeg', 0.9);
      link.click();
    };
    img.src = currentItem.image;
  }, [currentItem, currentIndex]);

  const reset = () => {
    setBatch([]);
    setCurrentIndex(0);
  };

  const removeItem = (id: string) => {
    setBatch(prev => {
      const newBatch = prev.filter(item => item.id !== id);
      if (currentIndex >= newBatch.length) {
        setCurrentIndex(Math.max(0, newBatch.length - 1));
      }
      return newBatch;
    });
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e0e0e0] font-mono p-4 md:p-8 selection:bg-[#FF4444] selection:text-white">
      {/* Header */}
      <header className="max-w-5xl mx-auto mb-12 flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[#333] pb-6">
        <div>
          <div className="flex items-center gap-2 text-[#FF4444] mb-2">
            <div className="w-2 h-2 rounded-full bg-[#FF4444] animate-pulse" />
            <span className="text-[10px] tracking-[0.2em] uppercase font-bold">System Active // MeterVision v1.0</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-white">
            METER<span className="text-[#FF4444]">OCR</span>
          </h1>
        </div>
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex bg-[#151619] border border-[#333] rounded-lg p-1">
            <button
              onClick={() => setEngine('gemini')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-[10px] uppercase tracking-widest transition-all ${
                engine === 'gemini' ? 'bg-[#FF4444] text-white' : 'text-[#666] hover:text-[#999]'
              }`}
            >
              <Sparkles className="w-3 h-3" />
              Gemini AI
            </button>
            <button
              onClick={() => setEngine('local')}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-[10px] uppercase tracking-widest transition-all ${
                engine === 'local' ? 'bg-[#FF4444] text-white' : 'text-[#666] hover:text-[#999]'
              }`}
            >
              <Cpu className="w-3 h-3" />
              Local Model
            </button>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-[#666] uppercase tracking-widest">Processing Engine</p>
            <p className="text-sm text-[#999]">{engine === 'gemini' ? 'Gemini Multimodal Vision' : 'On-Device CNN (TF.js)'}</p>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Area */}
        <div className="lg:col-span-7 space-y-6">
          <div className="relative aspect-video bg-[#151619] rounded-xl border border-[#333] overflow-hidden group flex items-center justify-center">
            {!currentItem ? (
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center gap-4 cursor-pointer hover:scale-105 transition-transform"
              >
                <div className="w-16 h-16 rounded-full bg-[#FF4444]/10 flex items-center justify-center text-[#FF4444] border border-[#FF4444]/20">
                  <Upload className="w-8 h-8" />
                </div>
                <p className="text-sm text-[#666] uppercase tracking-widest">Drop meter images or click to upload</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleImageUpload} 
                  accept="image/*" 
                  multiple
                  className="hidden" 
                />
              </div>
            ) : (
              <>
                <img 
                  ref={imgRef}
                  src={currentItem.image} 
                  alt="Meter Preview" 
                  className="w-full h-full object-contain"
                  referrerPolicy="no-referrer"
                  onLoad={updateImgRect}
                />
                
                {/* ROI Bounding Box Overlay */}
                {currentItem.result?.roi && !currentItem.isProcessing && imgDisplayRect && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="absolute border-2 border-[#FF4444] shadow-[0_0_20px_rgba(255,68,68,0.8)] z-20 pointer-events-none"
                    style={{
                      top: `${imgDisplayRect.top + (currentItem.result.roi[0] / 1000) * imgDisplayRect.height}%`,
                      left: `${imgDisplayRect.left + (currentItem.result.roi[1] / 1000) * imgDisplayRect.width}%`,
                      width: `${((currentItem.result.roi[3] - currentItem.result.roi[1]) / 1000) * imgDisplayRect.width}%`,
                      height: `${((currentItem.result.roi[2] - currentItem.result.roi[0]) / 1000) * imgDisplayRect.height}%`,
                    }}
                  >
                    <div className="absolute -top-6 left-0 bg-[#FF4444] text-white text-[10px] px-2 py-0.5 font-bold uppercase tracking-widest rounded-t-sm whitespace-nowrap">
                      ROI: {currentItem.result.digits}
                    </div>
                  </motion.div>
                )}

                <div className="absolute top-4 right-4 flex gap-2">
                  <button 
                    onClick={reset}
                    className="p-2 bg-black/50 backdrop-blur-md rounded-lg border border-white/10 hover:bg-black/80 transition-colors text-white"
                    title="Clear All"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
                
                {/* Scanning Animation Overlay */}
                {currentItem.isProcessing && (
                  <motion.div 
                    initial={{ top: 0 }}
                    animate={{ top: '100%' }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="absolute left-0 right-0 h-1 bg-[#FF4444] shadow-[0_0_15px_rgba(255,68,68,0.8)] z-10"
                  />
                )}
              </>
            )}
          </div>

          {/* Batch Thumbnails */}
          {batch.length > 0 && (
            <div className="bg-[#151619] border border-[#333] rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[10px] uppercase tracking-widest text-[#666] font-bold">Batch Queue ({batch.length})</h3>
                <button 
                  onClick={processBatch}
                  disabled={isBatchProcessing}
                  className="text-[10px] uppercase tracking-widest text-[#FF4444] hover:text-[#FF6666] flex items-center gap-1 transition-colors disabled:opacity-50"
                >
                  <Cpu className={`w-3 h-3 ${isBatchProcessing ? 'animate-spin' : ''}`} />
                  Process All
                </button>
              </div>
              <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                {batch.map((item, index) => (
                  <div 
                    key={item.id}
                    onClick={() => setCurrentIndex(index)}
                    className={`relative flex-shrink-0 w-20 h-20 rounded-lg border-2 cursor-pointer transition-all overflow-hidden group ${
                      currentIndex === index ? 'border-[#FF4444]' : 'border-[#333] hover:border-[#444]'
                    }`}
                  >
                    <img src={item.image} className="w-full h-full object-cover" />
                    {item.result && (
                      <div className="absolute inset-0 bg-green-500/20 flex items-center justify-center">
                        <CheckCircle2 className="w-6 h-6 text-green-500" />
                      </div>
                    )}
                    {item.isProcessing && (
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                        <RefreshCw className="w-6 h-6 text-[#FF4444] animate-spin" />
                      </div>
                    )}
                    <button 
                      onClick={(e) => { e.stopPropagation(); removeItem(item.id); }}
                      className="absolute top-1 right-1 p-0.5 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <RefreshCw className="w-2 h-2 text-white" />
                    </button>
                  </div>
                ))}
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-shrink-0 w-20 h-20 rounded-lg border-2 border-dashed border-[#333] flex flex-col items-center justify-center text-[#444] hover:border-[#FF4444] hover:text-[#FF4444] transition-all"
                >
                  <Upload className="w-6 h-6" />
                  <span className="text-[8px] uppercase tracking-widest mt-1">Add</span>
                </button>
              </div>
            </div>
          )}

          <div className="flex gap-4">
            <button
              disabled={!currentItem || currentItem.isProcessing || isBatchProcessing}
              onClick={() => processItem(currentIndex)}
              className={`flex-1 py-4 rounded-xl font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all ${
                !currentItem || currentItem.isProcessing || isBatchProcessing
                ? 'bg-[#222] text-[#444] cursor-not-allowed' 
                : 'bg-[#FF4444] text-white hover:bg-[#FF6666] active:scale-[0.98] shadow-[0_0_20px_rgba(255,68,68,0.2)]'
              }`}
            >
              {currentItem?.isProcessing ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Maximize2 className="w-5 h-5" />
                  Run Detection + OCR
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Results Area */}
        <div className="lg:col-span-5 space-y-6">
          <section className="bg-[#151619] rounded-xl border border-[#333] p-6 h-full flex flex-col">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-xs uppercase tracking-[0.3em] text-[#666] font-bold">Analysis Results</h2>
              <FileText className="w-4 h-4 text-[#444]" />
            </div>

            <div className="flex-1 space-y-8">
              <AnimatePresence mode="wait">
                {currentItem?.result ? (
                  <motion.div
                    key={currentItem.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-8"
                  >
                    {/* Digit Display */}
                    <div>
                      <label className="text-[10px] uppercase text-[#444] mb-2 block tracking-widest">
                        {engine === 'local' ? 'Predicted Digit' : 'Extracted Digits'}
                      </label>
                      <div className="bg-black border border-[#333] p-6 rounded-lg flex items-center justify-center">
                        <span className="text-6xl font-bold tracking-[0.2em] text-white tabular-nums">
                          {currentItem.result.digits}
                        </span>
                      </div>
                    </div>

                    {/* Metadata Grid */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#1a1b1e] p-4 rounded-lg border border-[#333]">
                        <p className="text-[10px] uppercase text-[#444] mb-1 tracking-widest">Meter Type</p>
                        <p className="text-sm font-bold text-[#ccc]">{currentItem.result.meterType}</p>
                      </div>
                      <div className="bg-[#1a1b1e] p-4 rounded-lg border border-[#333]">
                        <p className="text-[10px] uppercase text-[#444] mb-1 tracking-widest">Confidence</p>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${
                            (currentItem.result.confidence?.toLowerCase() || '') === 'high' ? 'bg-green-500' : 
                            (currentItem.result.confidence?.toLowerCase() || '') === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                          }`} />
                          <p className="text-sm font-bold text-[#ccc]">{currentItem.result.confidence || 'N/A'}</p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4">
                      {engine === 'gemini' && (
                        <button 
                          onClick={handleSave}
                          className="w-full py-3 border border-[#333] rounded-lg text-xs uppercase tracking-widest hover:bg-[#222] transition-colors flex items-center justify-center gap-2"
                        >
                          <Save className="w-4 h-4" />
                          Save Record
                        </button>
                      )}
                      {engine === 'local' && (
                        <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                          <p className="text-[10px] text-blue-400 uppercase tracking-widest leading-relaxed">
                            Testing local model performance. Note: The current local model is trained as a single-digit classifier (0-9).
                          </p>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ) : currentItem?.error ? (
                  <motion.div
                    key={`error-${currentItem.id}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center h-full text-center p-8"
                  >
                    <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
                    <p className="text-sm text-[#999]">{currentItem.error}</p>
                  </motion.div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center opacity-20">
                    <Camera className="w-16 h-16 mb-4" />
                    <p className="text-xs uppercase tracking-widest">Waiting for input...</p>
                  </div>
                )}
              </AnimatePresence>
            </div>

            {/* System Logs */}
            <div className="mt-auto pt-6 border-t border-[#222]">
              <div className="flex items-center justify-between text-[10px] text-[#444] uppercase tracking-widest mb-2">
                <span>System Logs</span>
                <span>Live</span>
              </div>
              <div className="bg-black/50 rounded p-3 font-mono text-[9px] text-[#555] h-24 overflow-y-auto space-y-1">
                <p>{`> [${new Date().toLocaleTimeString()}] System initialized`}</p>
                {batch.length > 0 && <p>{`> [${new Date().toLocaleTimeString()}] Batch queue: ${batch.length} items`}</p>}
                {currentItem?.isProcessing && <p className="text-[#FF4444]">{`> [${new Date().toLocaleTimeString()}] Processing: ${currentItem.name}`}</p>}
                {currentItem?.result && <p className="text-green-500">{`> [${new Date().toLocaleTimeString()}] Success: ${currentItem.result.digits}`}</p>}
                {currentItem?.error && <p className="text-red-500">{`> [${new Date().toLocaleTimeString()}] Error: ${currentItem.error}`}</p>}
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Hidden canvas for cropping */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Footer Decoration */}
      <footer className="max-w-5xl mx-auto mt-12 flex items-center justify-between text-[10px] text-[#333] uppercase tracking-[0.5em]">
        <span>Precision Vision Engine</span>
        <div className="flex gap-8">
          <span>ROI: Auto-Detect</span>
          <span>Mode: High-Accuracy</span>
        </div>
      </footer>
    </div>
  );
}
