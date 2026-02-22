/**
 * TextToSpeech Component - Browser-based TTS using Web Speech API.
 * Provides play/pause/stop controls, voice selection, speed, pitch, and volume.
 * Standalone component that can be embedded anywhere.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Volume2, VolumeX, Play, Pause, Square, SkipForward, SkipBack, Settings2, ChevronDown } from 'lucide-react';

export default function TextToSpeech({ text, label = 'Read Aloud', compact = false }) {
  const [speaking, setSpeaking] = useState(false);
  const [paused, setPaused] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [rate, setRate] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [volume, setVolume] = useState(1.0);
  const [showSettings, setShowSettings] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentWord, setCurrentWord] = useState('');
  const utteranceRef = useRef(null);
  const intervalRef = useRef(null);
  const totalCharsRef = useRef(0);
  const spokenCharsRef = useRef(0);

  // Load available voices
  useEffect(() => {
    const loadVoices = () => {
      const v = window.speechSynthesis?.getVoices() || [];
      setVoices(v);
      if (v.length > 0 && !selectedVoice) {
        const eng = v.find(voice => voice.lang.startsWith('en') && voice.default) ||
                    v.find(voice => voice.lang.startsWith('en')) ||
                    v[0];
        if (eng) setSelectedVoice(eng.name);
      }
    };
    loadVoices();
    window.speechSynthesis?.addEventListener('voiceschanged', loadVoices);
    return () => {
      window.speechSynthesis?.removeEventListener('voiceschanged', loadVoices);
      window.speechSynthesis?.cancel();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const cleanText = useCallback((t) => {
    if (!t) return '';
    // Remove markdown formatting, extra whitespace
    return t
      .replace(/[#*_~`>]/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }, []);

  const speak = useCallback(() => {
    if (!window.speechSynthesis || !text) return;

    window.speechSynthesis.cancel();
    const cleaned = cleanText(text);
    if (!cleaned) return;

    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = volume;

    const voice = voices.find(v => v.name === selectedVoice);
    if (voice) utterance.voice = voice;

    totalCharsRef.current = cleaned.length;
    spokenCharsRef.current = 0;

    utterance.onstart = () => {
      setSpeaking(true);
      setPaused(false);
      setProgress(0);
    };

    utterance.onboundary = (e) => {
      if (e.name === 'word') {
        spokenCharsRef.current = e.charIndex;
        setProgress(Math.round((e.charIndex / totalCharsRef.current) * 100));
        setCurrentWord(cleaned.substring(e.charIndex, e.charIndex + e.charLength));
      }
    };

    utterance.onend = () => {
      setSpeaking(false);
      setPaused(false);
      setProgress(100);
      setCurrentWord('');
      setTimeout(() => setProgress(0), 1500);
    };

    utterance.onerror = () => {
      setSpeaking(false);
      setPaused(false);
      setProgress(0);
    };

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  }, [text, rate, pitch, volume, selectedVoice, voices, cleanText]);

  const togglePause = () => {
    if (!window.speechSynthesis) return;
    if (paused) {
      window.speechSynthesis.resume();
      setPaused(false);
    } else {
      window.speechSynthesis.pause();
      setPaused(true);
    }
  };

  const stop = () => {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
    setPaused(false);
    setProgress(0);
    setCurrentWord('');
  };

  const changeRate = (delta) => {
    setRate(prev => Math.min(3, Math.max(0.25, +(prev + delta).toFixed(2))));
  };

  if (!window.speechSynthesis) {
    return null; // Browser doesn't support TTS
  }

  if (compact) {
    return (
      <div className="inline-flex items-center gap-1">
        {!speaking ? (
          <button
            onClick={speak}
            disabled={!text}
            className="p-1.5 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition disabled:opacity-30"
            title={label}
          >
            <Volume2 className="w-4 h-4" />
          </button>
        ) : (
          <>
            <button onClick={togglePause} className="p-1.5 text-primary-600 hover:bg-primary-50 rounded-lg transition" title={paused ? 'Resume' : 'Pause'}>
              {paused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            </button>
            <button onClick={stop} className="p-1.5 text-red-500 hover:bg-red-50 rounded-lg transition" title="Stop">
              <Square className="w-3.5 h-3.5" />
            </button>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Progress Bar */}
      {speaking && (
        <div className="h-1 bg-gray-100">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      <div className="p-4">
        {/* Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            {!speaking ? (
              <button
                onClick={speak}
                disabled={!text}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition disabled:opacity-50 text-sm font-medium"
              >
                <Volume2 className="w-4 h-4" />
                {label}
              </button>
            ) : (
              <>
                <button
                  onClick={togglePause}
                  className="p-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition"
                  title={paused ? 'Resume' : 'Pause'}
                >
                  {paused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                </button>
                <button
                  onClick={stop}
                  className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition"
                  title="Stop"
                >
                  <Square className="w-4 h-4" />
                </button>
              </>
            )}
          </div>

          {/* Speed Controls */}
          <div className="flex items-center gap-1 ml-2">
            <button onClick={() => changeRate(-0.25)} className="p-1 text-gray-400 hover:text-gray-600 rounded" title="Slower">
              <SkipBack className="w-3.5 h-3.5" />
            </button>
            <span className="text-xs font-mono text-gray-500 min-w-[40px] text-center">{rate}x</span>
            <button onClick={() => changeRate(0.25)} className="p-1 text-gray-400 hover:text-gray-600 rounded" title="Faster">
              <SkipForward className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Current Word */}
          {speaking && currentWord && (
            <span className="text-xs text-primary-600 font-medium truncate max-w-[120px] bg-primary-50 px-2 py-0.5 rounded">
              {currentWord}
            </span>
          )}

          {/* Progress */}
          {speaking && (
            <span className="text-xs text-gray-400 ml-auto">{progress}%</span>
          )}

          {/* Settings Toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition ml-auto ${showSettings ? 'bg-gray-100 text-gray-700' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-50'}`}
            title="Voice Settings"
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mt-3 pt-3 border-t border-gray-100 space-y-3">
            {/* Voice Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Voice</label>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
              >
                {voices.map(v => (
                  <option key={v.name} value={v.name}>
                    {v.name} ({v.lang})
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {/* Speed */}
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Speed: {rate}x</label>
                <input
                  type="range" min="0.25" max="3" step="0.25" value={rate}
                  onChange={(e) => setRate(parseFloat(e.target.value))}
                  className="w-full accent-primary-600"
                />
              </div>
              {/* Pitch */}
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Pitch: {pitch}</label>
                <input
                  type="range" min="0.5" max="2" step="0.1" value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                  className="w-full accent-primary-600"
                />
              </div>
              {/* Volume */}
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Volume: {Math.round(volume * 100)}%</label>
                <input
                  type="range" min="0" max="1" step="0.1" value={volume}
                  onChange={(e) => setVolume(parseFloat(e.target.value))}
                  className="w-full accent-primary-600"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
