/**
 * Study Mode Page - Flashcard-based studying from document content.
 * Generate flashcards, flip through them, track progress.
 * Includes TTS integration for reading cards aloud.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { documentsAPI } from '../services/api';
import {
  BookOpen, Loader, RotateCcw, ChevronLeft, ChevronRight,
  FileText, Sparkles, Shuffle, CheckCircle, XCircle,
  GraduationCap, Layers, Volume2
} from 'lucide-react';
import TextToSpeech from '../components/TextToSpeech';
import toast from 'react-hot-toast';

export default function StudyMode() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
  const [numCards, setNumCards] = useState(10);
  const [flashcards, setFlashcards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState(new Set());
  const [unknown, setUnknown] = useState(new Set());
  const [studyMode, setStudyMode] = useState('all'); // 'all', 'unknown', 'shuffle'

  useEffect(() => {
    documentsAPI.list().then((res) => {
      setDocuments(res.data.filter((d) => d.status === 'processed'));
      setLoading(false);
    }).catch(() => { setLoading(false); toast.error('Failed to load documents'); });
  }, []);

  const generateFlashcards = async () => {
    if (!selectedDoc) {
      toast.error('Select a document first');
      return;
    }
    setGenerating(true);
    setFlashcards([]);
    setCurrentIndex(0);
    setFlipped(false);
    setKnown(new Set());
    setUnknown(new Set());
    try {
      const res = await documentsAPI.generateFlashcards({
        document_id: parseInt(selectedDoc),
        num_cards: numCards,
      });
      if (res.data.flashcards?.length > 0) {
        setFlashcards(res.data.flashcards);
        toast.success(`Generated ${res.data.flashcards.length} flashcards`);
      } else {
        toast.error('Could not generate flashcards from this document');
      }
    } catch {
      toast.error('Failed to generate flashcards');
    } finally {
      setGenerating(false);
    }
  };

  const getActiveCards = useCallback(() => {
    if (studyMode === 'unknown') {
      return flashcards.filter((_, i) => unknown.has(i) || !known.has(i));
    }
    return flashcards;
  }, [flashcards, studyMode, known, unknown]);

  const activeCards = getActiveCards();

  const goNext = () => {
    setFlipped(false);
    if (studyMode === 'shuffle') {
      setCurrentIndex(Math.floor(Math.random() * activeCards.length));
    } else {
      setCurrentIndex((prev) => (prev + 1) % activeCards.length);
    }
  };

  const goPrev = () => {
    setFlipped(false);
    setCurrentIndex((prev) => (prev - 1 + activeCards.length) % activeCards.length);
  };

  const markKnown = () => {
    const realIndex = flashcards.indexOf(activeCards[currentIndex]);
    setKnown((prev) => new Set([...prev, realIndex]));
    setUnknown((prev) => { const n = new Set(prev); n.delete(realIndex); return n; });
    goNext();
  };

  const markUnknown = () => {
    const realIndex = flashcards.indexOf(activeCards[currentIndex]);
    setUnknown((prev) => new Set([...prev, realIndex]));
    setKnown((prev) => { const n = new Set(prev); n.delete(realIndex); return n; });
    goNext();
  };

  const resetStudy = () => {
    setKnown(new Set());
    setUnknown(new Set());
    setCurrentIndex(0);
    setFlipped(false);
    setStudyMode('all');
  };

  const currentCard = activeCards[currentIndex];
  const progress = flashcards.length > 0 ? Math.round((known.size / flashcards.length) * 100) : 0;

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <Loader className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <GraduationCap className="w-7 h-7 text-purple-500" />
          Study Mode
        </h1>
        <p className="text-gray-500 mt-1">Generate flashcards from your documents and study interactively</p>
      </div>

      {/* Generator Controls */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm mb-6">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs font-medium text-gray-500 mb-1">Select Document</label>
            <select
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
            >
              <option value="">Choose a document...</option>
              {documents.map((d) => (
                <option key={d.id} value={d.id}>{d.original_name}</option>
              ))}
            </select>
          </div>
          <div className="w-32">
            <label className="block text-xs font-medium text-gray-500 mb-1">Cards</label>
            <select
              value={numCards}
              onChange={(e) => setNumCards(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
            >
              {[5, 10, 15, 20, 25].map((n) => (
                <option key={n} value={n}>{n} cards</option>
              ))}
            </select>
          </div>
          <button
            onClick={generateFlashcards}
            disabled={generating || !selectedDoc}
            className="flex items-center gap-2 px-5 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition disabled:opacity-50 text-sm font-medium"
          >
            {generating ? <Loader className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Generate
          </button>
        </div>
      </div>

      {/* Flashcard Area */}
      {flashcards.length > 0 && (
        <div className="space-y-4">
          {/* Progress Bar */}
          <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-500">
                Progress: {known.size} / {flashcards.length} mastered
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-green-600">{known.size} known</span>
                <span className="text-xs text-red-500">{unknown.size} review</span>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-purple-500 to-green-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Study Mode Toggle */}
          <div className="flex items-center gap-2">
            {[
              { id: 'all', label: 'All Cards', icon: Layers },
              { id: 'unknown', label: 'Needs Review', icon: RotateCcw },
              { id: 'shuffle', label: 'Shuffle', icon: Shuffle },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => { setStudyMode(id); setCurrentIndex(0); setFlipped(false); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                  studyMode === id ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            ))}
            <button
              onClick={resetStudy}
              className="ml-auto text-xs text-gray-400 hover:text-gray-600 transition"
            >
              Reset Progress
            </button>
          </div>

          {/* Flashcard */}
          {activeCards.length > 0 && currentCard ? (
            <div className="relative">
              <div
                onClick={() => setFlipped(!flipped)}
                className={`bg-white rounded-2xl border-2 shadow-lg cursor-pointer transition-all duration-300 min-h-[280px] flex flex-col items-center justify-center p-8 select-none ${
                  flipped ? 'border-purple-300 bg-purple-50' : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                {/* Card Number */}
                <div className="absolute top-4 left-4 text-xs text-gray-400">
                  {currentIndex + 1} / {activeCards.length}
                </div>

                {/* Flip Indicator */}
                <div className="absolute top-4 right-4 text-xs text-gray-300">
                  {flipped ? 'ANSWER' : 'QUESTION'} — Click to flip
                </div>

                {/* Card Content */}
                <div className="text-center max-w-lg">
                  {!flipped ? (
                    <>
                      <BookOpen className="w-8 h-8 text-purple-300 mx-auto mb-4" />
                      <p className="text-lg font-medium text-gray-900 leading-relaxed">
                        {currentCard.front}
                      </p>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                      <p className="text-base text-gray-700 leading-relaxed">
                        {currentCard.back}
                      </p>
                    </>
                  )}
                </div>

                {/* TTS for current card */}
                <div className="absolute bottom-4 right-4">
                  <TextToSpeech
                    text={flipped ? currentCard.back : currentCard.front}
                    compact
                  />
                </div>
              </div>

              {/* Navigation & Actions */}
              <div className="flex items-center justify-between mt-4">
                <button
                  onClick={goPrev}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition"
                >
                  <ChevronLeft className="w-6 h-6" />
                </button>

                <div className="flex items-center gap-3">
                  <button
                    onClick={markUnknown}
                    className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition text-sm font-medium"
                  >
                    <XCircle className="w-4 h-4" />
                    Review Again
                  </button>
                  <button
                    onClick={markKnown}
                    className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-600 rounded-lg hover:bg-green-100 transition text-sm font-medium"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Got It!
                  </button>
                </div>

                <button
                  onClick={goNext}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition"
                >
                  <ChevronRight className="w-6 h-6" />
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-green-50 rounded-xl border border-green-200 p-8 text-center">
              <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-green-700">All cards mastered!</h3>
              <p className="text-sm text-green-500 mt-1">
                Switch to "All Cards" or generate new flashcards.
              </p>
            </div>
          )}

          {/* Full Flashcard TTS */}
          {currentCard && (
            <TextToSpeech
              text={`Question: ${currentCard.front}. Answer: ${currentCard.back}`}
              label="Read Full Card"
            />
          )}
        </div>
      )}

      {/* Empty State */}
      {flashcards.length === 0 && !generating && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">
          <GraduationCap className="w-16 h-16 text-gray-200 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600">Ready to Study?</h3>
          <p className="text-sm text-gray-400 mt-1">
            Select a document and generate flashcards to start studying
          </p>
        </div>
      )}
    </div>
  );
}
