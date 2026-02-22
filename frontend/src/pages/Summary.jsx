/**
 * Summary Page - Multi-document comparison and AI summarization.
 * Compare answers across multiple documents side-by-side.
 */

import React, { useState, useEffect } from 'react';
import { chatAPI, documentsAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import {
  Search, FileText, GitCompare, Sparkles, Loader, AlertCircle
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function Summary() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const { user } = useAuth();

  useEffect(() => {
    documentsAPI.list().then((res) => {
      setDocuments(res.data.filter((d) => d.status === 'processed'));
    });
  }, []);

  const toggleDoc = (docId) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleCompare = async () => {
    if (!query.trim()) {
      toast.error('Please enter a question');
      return;
    }
    if (selectedDocs.length < 2) {
      toast.error('Select at least 2 documents to compare');
      return;
    }

    setLoading(true);
    setComparisonResult(null);

    try {
      const res = await chatAPI.compare({
        query: query.trim(),
        document_ids: selectedDocs,
      });
      setComparisonResult(res.data);
    } catch (err) {
      toast.error('Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (conf) => {
    if (conf > 0.7) return 'text-green-600 bg-green-50';
    if (conf > 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Compare & Summarize</h1>
        <p className="text-gray-500 mt-1">Compare answers across multiple documents</p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6 shadow-sm">
        {/* Document Selection */}
        <div className="mb-4">
          <p className="text-sm font-medium text-gray-700 mb-2">Select Documents to Compare</p>
          <div className="flex flex-wrap gap-2">
            {documents.map((doc) => (
              <button
                key={doc.id}
                onClick={() => toggleDoc(doc.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition border ${
                  selectedDocs.includes(doc.id)
                    ? 'bg-primary-50 border-primary-300 text-primary-700'
                    : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
                }`}
              >
                <FileText className="w-4 h-4" />
                <span className="truncate max-w-[200px]">{doc.original_name}</span>
              </button>
            ))}
            {documents.length === 0 && (
              <p className="text-sm text-gray-400">No processed documents available</p>
            )}
          </div>
        </div>

        {/* Query Input */}
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
            placeholder="Enter a question to compare across documents..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
          />
          <button
            onClick={handleCompare}
            disabled={loading || selectedDocs.length < 2}
            className="flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition disabled:opacity-50"
          >
            {loading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              <GitCompare className="w-5 h-5" />
            )}
            Compare
          </button>
        </div>
      </div>

      {/* Comparison Results */}
      {comparisonResult && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold text-gray-900">Comparison Results</h2>
            <span className="text-sm text-gray-500">
              ({comparisonResult.document_count} documents)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {comparisonResult.comparisons?.map((comp, i) => (
              <div
                key={i}
                className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm"
              >
                {/* Document Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-red-500" />
                    <span className="font-medium text-gray-900 text-sm truncate">
                      {comp.filename}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full font-medium ${getConfidenceColor(comp.confidence)}`}>
                    {(comp.confidence * 100).toFixed(0)}% confident
                  </span>
                </div>

                {/* Answer */}
                <div className="markdown-content text-sm text-gray-700 mb-3">
                  <ReactMarkdown>{comp.answer}</ReactMarkdown>
                </div>

                {/* Sources */}
                {comp.sources?.length > 0 && (
                  <div className="border-t border-gray-100 pt-3">
                    <p className="text-xs text-gray-500 mb-1">Sources:</p>
                    {comp.sources.slice(0, 3).map((src, j) => (
                      <div key={j} className="text-xs text-gray-400 flex items-center gap-1">
                        <span>Page {src.page}</span>
                        <span className="text-gray-300">|</span>
                        <span>Score: {(src.score * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {comparisonResult.comparisons?.length === 0 && (
            <div className="text-center py-8">
              <AlertCircle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">No results found for this comparison</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
