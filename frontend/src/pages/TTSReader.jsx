/**
 * TTS Reader Page - Dedicated text-to-speech document reader.
 * Read entire documents or specific pages aloud with full TTS controls.
 * Completely standalone feature, separate from Chat or Insights.
 */

import React, { useState, useEffect } from 'react';
import { documentsAPI } from '../services/api';
import {
  FileText, Loader, Headphones, BookOpen, ChevronLeft, ChevronRight,
  Volume2, StickyNote, Trash2, Plus, MessageSquare
} from 'lucide-react';
import TextToSpeech from '../components/TextToSpeech';
import toast from 'react-hot-toast';

export default function TTSReader() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [textLoading, setTextLoading] = useState(false);
  const [ttsData, setTtsData] = useState(null);
  const [currentPage, setCurrentPage] = useState(null); // null = full doc
  const [maxChars, setMaxChars] = useState(5000);
  // Annotations
  const [annotations, setAnnotations] = useState([]);
  const [annotationText, setAnnotationText] = useState('');
  const [showAnnotations, setShowAnnotations] = useState(false);

  useEffect(() => {
    documentsAPI.list().then((res) => {
      setDocuments(res.data.filter((d) => d.status === 'processed'));
      setLoading(false);
    }).catch(() => { setLoading(false); toast.error('Failed to load documents'); });
  }, []);

  const loadText = async (doc, page = null) => {
    setSelectedDoc(doc);
    setCurrentPage(page);
    setTextLoading(true);
    try {
      const res = await documentsAPI.ttsText(doc.id, page, maxChars);
      setTtsData(res.data);
    } catch {
      toast.error('Failed to load document text');
    } finally {
      setTextLoading(false);
    }
  };

  const loadAnnotations = async (doc) => {
    try {
      const res = await documentsAPI.annotations(doc.id);
      setAnnotations(res.data);
    } catch {
      // Annotations might not exist yet
      setAnnotations([]);
    }
  };

  const addAnnotation = async () => {
    if (!annotationText.trim() || !selectedDoc) return;
    try {
      await documentsAPI.addAnnotation(selectedDoc.id, {
        page_number: currentPage || 1,
        content: annotationText.trim(),
        color: 'yellow',
      });
      setAnnotationText('');
      toast.success('Note added');
      loadAnnotations(selectedDoc);
    } catch {
      toast.error('Failed to add note');
    }
  };

  const deleteAnnotation = async (annId) => {
    if (!selectedDoc) return;
    try {
      await documentsAPI.deleteAnnotation(selectedDoc.id, annId);
      toast.success('Note deleted');
      loadAnnotations(selectedDoc);
    } catch {
      toast.error('Failed to delete note');
    }
  };

  const handleSelectDoc = (doc) => {
    loadText(doc);
    loadAnnotations(doc);
  };

  const goToPage = (page) => {
    if (selectedDoc) {
      loadText(selectedDoc, page);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <Loader className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Headphones className="w-7 h-7 text-teal-500" />
          Document Reader
        </h1>
        <p className="text-gray-500 mt-1">Listen to your documents with text-to-speech and add notes</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Document List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="p-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-900 text-sm flex items-center gap-2">
                <FileText className="w-4 h-4 text-gray-400" />
                Documents
              </h3>
            </div>
            <div className="max-h-[600px] overflow-y-auto">
              {documents.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => handleSelectDoc(doc)}
                  className={`w-full flex items-center gap-3 px-4 py-3 text-left transition border-b border-gray-100 ${
                    selectedDoc?.id === doc.id ? 'bg-teal-50' : 'hover:bg-gray-50'
                  }`}
                >
                  <FileText className="w-4 h-4 text-red-500 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900 truncate">{doc.original_name}</p>
                    <p className="text-xs text-gray-400">{doc.page_count || '?'} pages</p>
                  </div>
                </button>
              ))}
              {documents.length === 0 && (
                <p className="text-sm text-gray-400 text-center py-8">No processed documents</p>
              )}
            </div>
          </div>
        </div>

        {/* Reader Area */}
        <div className="lg:col-span-3 space-y-4">
          {textLoading ? (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex items-center justify-center py-20">
              <div className="text-center">
                <Loader className="w-8 h-8 text-teal-500 animate-spin mx-auto mb-3" />
                <p className="text-sm text-gray-500">Loading document text...</p>
              </div>
            </div>
          ) : ttsData ? (
            <>
              {/* Document Header */}
              <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">{ttsData.filename}</h3>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {ttsData.total_chars.toLocaleString()} characters
                      {currentPage ? ` • Page ${currentPage}` : ` • Full document`}
                      {ttsData.page_count && ` • ${ttsData.page_count} total pages`}
                    </p>
                  </div>

                  {/* Page Navigation */}
                  {ttsData.page_count && ttsData.page_count > 1 && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => goToPage(null)}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                          currentPage === null ? 'bg-teal-100 text-teal-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                      >
                        All
                      </button>
                      <button
                        onClick={() => goToPage(Math.max(1, (currentPage || 1) - 1))}
                        disabled={currentPage === 1}
                        className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <select
                        value={currentPage || ''}
                        onChange={(e) => goToPage(e.target.value ? parseInt(e.target.value) : null)}
                        className="px-2 py-1 border border-gray-300 rounded text-xs focus:ring-2 focus:ring-teal-500 outline-none"
                      >
                        <option value="">All Pages</option>
                        {Array.from({ length: ttsData.page_count }, (_, i) => (
                          <option key={i + 1} value={i + 1}>Page {i + 1}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => goToPage(Math.min(ttsData.page_count, (currentPage || 0) + 1))}
                        disabled={currentPage === ttsData.page_count}
                        className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-30"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* TTS Player */}
              <TextToSpeech text={ttsData.text} label="Read Document" />

              {/* Text Preview */}
              <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <h4 className="text-sm font-medium text-gray-500 mb-3 flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Text Content
                </h4>
                <div className="max-h-[400px] overflow-y-auto">
                  <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {ttsData.text || 'No text content available for this page.'}
                  </p>
                </div>
              </div>

              {/* Annotations Section */}
              <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <button
                  onClick={() => setShowAnnotations(!showAnnotations)}
                  className="flex items-center justify-between w-full"
                >
                  <h4 className="text-sm font-medium text-gray-900 flex items-center gap-2">
                    <StickyNote className="w-4 h-4 text-amber-500" />
                    Notes & Annotations
                    {annotations.length > 0 && (
                      <span className="bg-amber-100 text-amber-700 text-xs px-1.5 py-0.5 rounded-full">{annotations.length}</span>
                    )}
                  </h4>
                  <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${showAnnotations ? 'rotate-90' : ''}`} />
                </button>

                {showAnnotations && (
                  <div className="mt-3 space-y-3">
                    {/* Add Note */}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={annotationText}
                        onChange={(e) => setAnnotationText(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && addAnnotation()}
                        placeholder={`Add a note for page ${currentPage || 1}...`}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-amber-500 focus:border-transparent outline-none"
                      />
                      <button
                        onClick={addAnnotation}
                        disabled={!annotationText.trim()}
                        className="px-3 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition disabled:opacity-50 text-sm"
                      >
                        <Plus className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Existing Notes */}
                    {annotations.length === 0 ? (
                      <p className="text-xs text-gray-400 text-center py-3">No notes yet</p>
                    ) : (
                      <div className="space-y-2 max-h-[200px] overflow-y-auto">
                        {annotations.map((ann) => (
                          <div key={ann.id} className="flex items-start gap-2 p-2 bg-amber-50 rounded-lg border border-amber-100">
                            <MessageSquare className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-gray-700">{ann.content}</p>
                              <p className="text-[10px] text-gray-400 mt-0.5">Page {ann.page_number}</p>
                            </div>
                            <button
                              onClick={() => deleteAnnotation(ann.id)}
                              className="p-1 text-gray-300 hover:text-red-500 transition shrink-0"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex items-center justify-center py-20">
              <div className="text-center">
                <Headphones className="w-16 h-16 text-gray-200 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-600">Select a document</h3>
                <p className="text-sm text-gray-400 mt-1">Choose a document to listen to with text-to-speech</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
