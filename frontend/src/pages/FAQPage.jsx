/**
 * FAQ Page - Auto-generate and manage FAQs from documents.
 * Select documents, generate Q&A pairs, browse saved FAQs.
 */

import React, { useState, useEffect } from 'react';
import { documentsAPI, faqAPI } from '../services/api';
import ReactMarkdown from 'react-markdown';
import {
  FileText, Loader, HelpCircle, Sparkles, Trash2,
  ChevronDown, ChevronUp, RefreshCw, Plus, BookOpen
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function FAQPage() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [faqs, setFaqs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [numQuestions, setNumQuestions] = useState(5);
  const [expandedFaq, setExpandedFaq] = useState({});
  const [filterDoc, setFilterDoc] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [docRes, faqRes] = await Promise.all([
          documentsAPI.list(),
          faqAPI.list(),
        ]);
        setDocuments(docRes.data.filter((d) => d.status === 'processed'));
        setFaqs(faqRes.data.faqs || []);
      } catch {
        toast.error('Failed to load data');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const toggleDoc = (docId) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleGenerate = async () => {
    if (selectedDocs.length === 0) {
      toast.error('Select at least one document');
      return;
    }
    setGenerating(true);
    try {
      const res = await faqAPI.generate({
        document_ids: selectedDocs,
        num_questions: numQuestions,
      });
      const newFaqs = res.data.faqs || [];
      setFaqs((prev) => [...newFaqs, ...prev]);
      toast.success(`Generated ${newFaqs.length} FAQs!`);
    } catch {
      toast.error('FAQ generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await faqAPI.delete(id);
      setFaqs((prev) => prev.filter((f) => f.id !== id));
      toast.success('FAQ deleted');
    } catch {
      toast.error('Failed to delete FAQ');
    }
  };

  const toggleExpand = (id) => {
    setExpandedFaq((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredFaqs = filterDoc
    ? faqs.filter((f) => f.document_id === filterDoc)
    : faqs;

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <Loader className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">FAQ Generator</h1>
        <p className="text-gray-500 mt-1">Auto-generate frequently asked questions from your documents</p>
      </div>

      {/* Generator Controls */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-amber-500" />
          Generate New FAQs
        </h2>

        {/* Document Selection */}
        <div className="mb-4">
          <p className="text-sm font-medium text-gray-700 mb-2">Select Documents</p>
          <div className="flex flex-wrap gap-2">
            {documents.map((doc) => (
              <button
                key={doc.id}
                onClick={() => toggleDoc(doc.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition border ${
                  selectedDocs.includes(doc.id)
                    ? 'bg-amber-50 border-amber-300 text-amber-700'
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

        {/* Options */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Questions per document:</label>
            <select
              value={numQuestions}
              onChange={(e) => setNumQuestions(Number(e.target.value))}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-amber-500 outline-none"
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleGenerate}
            disabled={generating || selectedDocs.length === 0}
            className="flex items-center gap-2 px-6 py-2.5 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition disabled:opacity-50 text-sm font-medium"
          >
            {generating ? (
              <><Loader className="w-4 h-4 animate-spin" /> Generating...</>
            ) : (
              <><Plus className="w-4 h-4" /> Generate FAQs</>
            )}
          </button>
        </div>
      </div>

      {/* Filter by Document */}
      {faqs.length > 0 && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-sm text-gray-500">Filter:</span>
          <button
            onClick={() => setFilterDoc(null)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition ${
              !filterDoc ? 'bg-primary-100 text-primary-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            All ({faqs.length})
          </button>
          {[...new Set(faqs.map((f) => f.document_id))].map((docId) => {
            const doc = documents.find((d) => d.id === docId);
            const count = faqs.filter((f) => f.document_id === docId).length;
            return (
              <button
                key={docId}
                onClick={() => setFilterDoc(docId === filterDoc ? null : docId)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition ${
                  filterDoc === docId ? 'bg-primary-100 text-primary-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {doc?.original_name?.slice(0, 20) || `Doc ${docId}`} ({count})
              </button>
            );
          })}
        </div>
      )}

      {/* FAQ List */}
      {filteredFaqs.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm py-16 text-center">
          <HelpCircle className="w-16 h-16 text-gray-200 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600">No FAQs yet</h3>
          <p className="text-sm text-gray-400 mt-1">Select documents and generate FAQs to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredFaqs.map((faq, i) => (
            <div key={faq.id || i} className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <button
                onClick={() => toggleExpand(faq.id || i)}
                className="w-full flex items-center gap-3 p-4 text-left hover:bg-gray-50 transition"
              >
                <HelpCircle className="w-5 h-5 text-amber-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">{faq.question}</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    <FileText className="w-3 h-3 inline mr-1" />
                    {faq.filename || `Document ${faq.document_id}`}
                    {faq.page_numbers?.length > 0 && ` · Pages ${faq.page_numbers.join(', ')}`}
                  </p>
                </div>
                {expandedFaq[faq.id || i] ? (
                  <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
              </button>
              {expandedFaq[faq.id || i] && (
                <div className="px-4 pb-4 pt-0">
                  <div className="bg-amber-50 rounded-lg p-4 border border-amber-100">
                    <div className="text-sm text-gray-700 markdown-content">
                      <ReactMarkdown>{faq.answer}</ReactMarkdown>
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-3">
                    <span className="text-xs text-gray-400">
                      {faq.created_at ? new Date(faq.created_at).toLocaleDateString() : ''}
                    </span>
                    {faq.id && (
                      <button
                        onClick={() => handleDelete(faq.id)}
                        className="p-1.5 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50 transition"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
