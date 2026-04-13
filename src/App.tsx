/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import MeterOCR from './components/MeterOCR';
import TrainingLab from './components/TrainingLab';
import { Camera, Cpu } from 'lucide-react';

export default function App() {
  const [view, setView] = useState<'scanner' | 'modeling'>('scanner');
  const [trainedModel, setTrainedModel] = useState<tf.LayersModel | null>(null);

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Global Navigation */}
      <nav className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 bg-black/80 backdrop-blur-xl border border-white/10 rounded-full p-1.5 flex gap-1 shadow-2xl">
        <button
          onClick={() => setView('scanner')}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-xs font-bold uppercase tracking-widest transition-all ${
            view === 'scanner' 
            ? 'bg-[#FF4444] text-white' 
            : 'text-[#666] hover:text-white hover:bg-white/5'
          }`}
        >
          <Camera className="w-4 h-4" />
          Scanner
        </button>
        <button
          onClick={() => setView('modeling')}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-xs font-bold uppercase tracking-widest transition-all ${
            view === 'modeling' 
            ? 'bg-[#FF4444] text-white' 
            : 'text-[#666] hover:text-white hover:bg-white/5'
          }`}
        >
          <Cpu className="w-4 h-4" />
          Modeling
        </button>
      </nav>

      <main className="pb-24">
        {view === 'scanner' ? (
          <MeterOCR trainedModel={trainedModel} />
        ) : (
          <TrainingLab onModelTrained={setTrainedModel} />
        )}
      </main>
    </div>
  );
}
