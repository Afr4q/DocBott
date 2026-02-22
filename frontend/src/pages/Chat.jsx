/**
 * Chat Page - Interactive Q&A with documents.
 * Features: role-based answers, source display, confidence scores,
 * AI vs PDF comparison, question suggestions, feedback,
 * export chat, text-to-speech, keyboard shortcuts.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { chatAPI, documentsAPI, feedbackAPI, bookmarksAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import {
  Send, FileText, Star, ThumbsUp, ThumbsDown,
  ChevronDown, Sparkles, BookOpen, Loader, X, MessageSquare,
  Download, Volume2, VolumeX, Keyboard, Copy, Check, Bookmark, BookmarkCheck,
  Brain, FileSearch, Layers, Lightbulb, Shield, Highlighter, Languages,
  Info, FileJson, FileType, Globe
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [showSources, setShowSources] = useState({});
  const [showComparison, setShowComparison] = useState({});
  const [isSpeaking, setIsSpeaking] = useState(null);
  const [copiedId, setCopiedId] = useState(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [bookmarkedIds, setBookmarkedIds] = useState(new Set());
  const [answerMode, setAnswerMode] = useState('pdf'); // 'pdf' | 'ai' | 'both'
  const [showRelated, setShowRelated] = useState({});
  const [relatedContent, setRelatedContent] = useState({});
  const [relatedLoading, setRelatedLoading] = useState({});
  const [factChecks, setFactChecks] = useState({});
  const [factCheckLoading, setFactCheckLoading] = useState({});
  const [showFactCheck, setShowFactCheck] = useState({});
  const [highlights, setHighlights] = useState({});
  const [highlightLoading, setHighlightLoading] = useState({});
  const [showHighlights, setShowHighlights] = useState({});
  const [confidenceExplain, setConfidenceExplain] = useState({});
  const [showConfExplain, setShowConfExplain] = useState({});
  const [translations, setTranslations] = useState({});
  const [translating, setTranslating] = useState({});
  const [showExportMenu, setShowExportMenu] = useState(false);
  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const { user } = useAuth();

  // Load documents
  useEffect(() => {
    documentsAPI.list().then((res) => {
      const processed = res.data.filter((d) => d.status === 'processed');
      setDocuments(processed);
    });
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const query = input.trim();
    setInput('');

    // Add user message
    const userMsg = { role: 'user', content: query, id: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await chatAPI.query({
        query,
        document_ids: selectedDocs.length > 0 ? selectedDocs : undefined,
        session_id: sessionId,
        enable_ai_summary: true,
        top_k: 5,
        answer_mode: answerMode,
      });

      const data = res.data;
      if (!sessionId && data.session_id) setSessionId(data.session_id);

      // Add assistant message
      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        id: Date.now() + 1,
        query: query,
        confidence: data.confidence,
        sources: data.sources,
        reasoning: data.reasoning,
        ai_summary: data.ai_summary,
        direct_ai_answer: data.direct_ai_answer,
        answer_mode: data.answer_mode,
        comparison: data.comparison,
        suggested_questions: data.suggested_questions,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      toast.error('Failed to get answer');
      const errMsg = {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
        id: Date.now() + 1,
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestedQuestion = (question) => {
    setInput(question);
  };

  const handleFeedback = async (msgId, helpful) => {
    try {
      await feedbackAPI.submit({
        rating: helpful ? 5 : 1,
        helpful,
        query: messages.find((m) => m.id === msgId - 1)?.content || '',
      });
      toast.success(helpful ? 'Thanks for the feedback!' : 'We\'ll improve!');
    } catch {
      toast.error('Failed to submit feedback');
    }
  };

  const toggleDocSelection = (docId) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const toggleSources = (msgId) => {
    setShowSources((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  const toggleComparison = (msgId) => {
    setShowComparison((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  // ── Feature: Export chat as Markdown ──
  const exportChat = () => {
    if (messages.length === 0) return toast.error('No messages to export');
    let md = `# DocBott Chat Export\n_Exported: ${new Date().toLocaleString()}_\n\n---\n\n`;
    messages.forEach((msg) => {
      if (msg.role === 'user') {
        md += `**You:** ${msg.content}\n\n`;
      } else {
        md += `**DocBott:** ${msg.content}\n\n`;
        if (msg.confidence) md += `> Confidence: ${(msg.confidence * 100).toFixed(0)}%\n\n`;
        if (msg.sources?.length) {
          md += `**Sources:**\n`;
          msg.sources.forEach((s, i) => {
            md += `- ${s.file || 'Document'}, Page ${s.page} (Score: ${(s.score * 100).toFixed(0)}%)\n`;
          });
          md += '\n';
        }
        md += '---\n\n';
      }
    });
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `docbott-chat-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Chat exported!');
  };

  // ── Feature: Text-to-Speech ──
  const speak = (msgId, text) => {
    if (isSpeaking === msgId) {
      window.speechSynthesis.cancel();
      setIsSpeaking(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.onend = () => setIsSpeaking(null);
    utterance.onerror = () => setIsSpeaking(null);
    setIsSpeaking(msgId);
    window.speechSynthesis.speak(utterance);
  };

  // ── Feature: Copy answer ──
  const copyAnswer = (msgId, text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(msgId);
      setTimeout(() => setCopiedId(null), 2000);
      toast.success('Copied to clipboard');
    });
  };

  // ── Feature: Bookmark Answer ──
  const handleBookmark = async (msg) => {
    if (bookmarkedIds.has(msg.id)) return;
    try {
      await bookmarksAPI.create({
        query: msg.query || '',
        answer: msg.content,
        confidence: msg.confidence,
        sources: msg.sources?.map((s) => s.file).join(', ') || '',
        note: '',
      });
      setBookmarkedIds((prev) => new Set(prev).add(msg.id));
      toast.success('Bookmarked!');
    } catch {
      toast.error('Failed to bookmark');
    }
  };

  // ── Feature: Related Content from PDF ──
  const fetchRelated = async (msg) => {
    if (relatedContent[msg.id]) {
      setShowRelated((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }));
      return;
    }
    setRelatedLoading((prev) => ({ ...prev, [msg.id]: true }));
    setShowRelated((prev) => ({ ...prev, [msg.id]: true }));
    try {
      const res = await chatAPI.related({
        query: msg.query || '',
        document_ids: selectedDocs.length > 0 ? selectedDocs : undefined,
        answer_text: msg.content,
        top_k: 4,
      });
      setRelatedContent((prev) => ({ ...prev, [msg.id]: res.data.related || [] }));
    } catch {
      toast.error('Failed to load related content');
      setShowRelated((prev) => ({ ...prev, [msg.id]: false }));
    } finally {
      setRelatedLoading((prev) => ({ ...prev, [msg.id]: false }));
    }
  };

  // ── Feature: Fact-Check Answer ──
  const handleFactCheck = async (msg) => {
    if (factChecks[msg.id]) {
      setShowFactCheck((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }));
      return;
    }
    setFactCheckLoading((prev) => ({ ...prev, [msg.id]: true }));
    setShowFactCheck((prev) => ({ ...prev, [msg.id]: true }));
    try {
      const res = await chatAPI.factCheck({
        answer: msg.content,
        query: msg.query || '',
        sources: msg.sources,
      });
      setFactChecks((prev) => ({ ...prev, [msg.id]: res.data }));
    } catch {
      toast.error('Fact-check failed');
      setShowFactCheck((prev) => ({ ...prev, [msg.id]: false }));
    } finally {
      setFactCheckLoading((prev) => ({ ...prev, [msg.id]: false }));
    }
  };

  // ── Feature: PDF Highlighting ──
  const handleHighlight = async (msg) => {
    if (highlights[msg.id]) {
      setShowHighlights((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }));
      return;
    }
    if (!selectedDocs.length) {
      toast.error('Select a document first');
      return;
    }
    setHighlightLoading((prev) => ({ ...prev, [msg.id]: true }));
    setShowHighlights((prev) => ({ ...prev, [msg.id]: true }));
    try {
      const res = await chatAPI.highlight({
        document_id: selectedDocs[0],
        query: msg.query || msg.content,
      });
      const items = (res.data.highlights || []).map(h => ({ ...h, file: res.data.filename || 'Document' }));
      setHighlights((prev) => ({ ...prev, [msg.id]: items }));
    } catch {
      toast.error('Highlighting failed');
      setShowHighlights((prev) => ({ ...prev, [msg.id]: false }));
    } finally {
      setHighlightLoading((prev) => ({ ...prev, [msg.id]: false }));
    }
  };

  // ── Feature: Confidence Explainer ──
  const handleExplainConfidence = async (msg) => {
    if (confidenceExplain[msg.id]) {
      setShowConfExplain((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }));
      return;
    }
    try {
      const res = await chatAPI.explainConfidence({
        answer: msg.content,
        query: msg.query || '',
        sources: msg.sources,
      });
      setConfidenceExplain((prev) => ({ ...prev, [msg.id]: res.data }));
      setShowConfExplain((prev) => ({ ...prev, [msg.id]: true }));
    } catch {
      toast.error('Failed to explain confidence');
    }
  };

  // ── Feature: Translate Answer ──
  const handleTranslate = async (msg, targetLang) => {
    const key = `${msg.id}_${targetLang}`;
    if (translations[key]) return;
    setTranslating((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await chatAPI.translate({
        text: msg.content,
        target_lang: targetLang,
      });
      setTranslations((prev) => ({ ...prev, [key]: res.data }));
    } catch {
      toast.error('Translation failed');
    } finally {
      setTranslating((prev) => ({ ...prev, [key]: false }));
    }
  };

  // ── Feature: Multi-format Export ──
  const handleExportFormat = async (format) => {
    setShowExportMenu(false);
    if (messages.length === 0) return toast.error('No messages to export');
    try {
      const res = await chatAPI.exportChat({
        format,
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content,
          confidence: m.confidence,
          sources: m.sources,
        })),
      });
      const { content, filename } = res.data;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch {
      toast.error('Export failed');
    }
  };

  // ── Feature: Keyboard Shortcuts ──
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+/ — Focus input
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        inputRef.current?.focus();
      }
      // Ctrl+E — Export chat
      if (e.ctrlKey && e.key === 'e') {
        e.preventDefault();
        exportChat();
      }
      // Ctrl+K — Show shortcuts
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        setShowShortcuts((prev) => !prev);
      }
      // Escape — Close modals
      if (e.key === 'Escape') {
        setShowShortcuts(false);
        window.speechSynthesis.cancel();
        setIsSpeaking(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [messages]);

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Sidebar - Document Selection */}
      <div className="w-72 border-r border-gray-200 bg-white flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h3 className="font-semibold text-gray-900 text-sm">Documents</h3>
          <p className="text-xs text-gray-500 mt-0.5">Select documents to search</p>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {documents.length === 0 ? (
            <p className="text-sm text-gray-400 text-center mt-4">No documents available</p>
          ) : (
            documents.map((doc) => (
              <label
                key={doc.id}
                className={`flex items-center gap-2 p-2.5 rounded-lg cursor-pointer transition text-sm ${
                  selectedDocs.includes(doc.id)
                    ? 'bg-primary-50 border border-primary-200'
                    : 'hover:bg-gray-50 border border-transparent'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedDocs.includes(doc.id)}
                  onChange={() => toggleDocSelection(doc.id)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <FileText className="w-4 h-4 text-red-500 shrink-0" />
                <span className="truncate">{doc.original_name}</span>
              </label>
            ))
          )}
        </div>
        {selectedDocs.length > 0 && (
          <div className="p-3 border-t border-gray-200">
            <button
              onClick={() => setSelectedDocs([])}
              className="w-full text-xs text-gray-500 hover:text-gray-700 py-1"
            >
              Clear selection ({selectedDocs.length})
            </button>
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <MessageSquare className="w-16 h-16 text-gray-200 mb-4" />
              <h2 className="text-xl font-semibold text-gray-700">Ask anything about your documents</h2>
              <p className="text-gray-400 mt-2 max-w-md">
                Select documents from the sidebar and type your question below.
                Answers are grounded in your uploaded PDFs.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl ${msg.role === 'user' ? 'order-1' : ''}`}>
                {/* Message Bubble */}
                <div className={msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}>
                  <div className="markdown-content">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>

                {/* Assistant extras */}
                {msg.role === 'assistant' && (msg.confidence !== undefined || msg.answer_mode === 'ai') && (
                  <div className="mt-3 space-y-2">
                    {/* Answer Mode Badge & Confidence */}
                    <div className="flex items-center gap-3 text-xs text-gray-500 flex-wrap">
                      {msg.answer_mode && (
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                          msg.answer_mode === 'ai' ? 'bg-purple-100 text-purple-700' :
                          msg.answer_mode === 'both' ? 'bg-indigo-100 text-indigo-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {msg.answer_mode === 'ai' ? <><Brain className="w-3 h-3" /> AI Answer</> :
                           msg.answer_mode === 'both' ? <><Layers className="w-3 h-3" /> PDF + AI</> :
                           <><FileSearch className="w-3 h-3" /> PDF Answer</>}
                        </span>
                      )}
                      {msg.confidence > 0 && (
                        <span className="flex items-center gap-1">
                          <div className={`w-2 h-2 rounded-full ${msg.confidence > 0.7 ? 'bg-green-500' : msg.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`} />
                          Confidence: {(msg.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                      <span>Role: {user?.role}</span>
                    </div>

                    {/* Direct AI Answer (shown in "both" mode alongside PDF answer) */}
                    {msg.answer_mode === 'both' && msg.direct_ai_answer && (
                      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-3 border border-purple-200">
                        <p className="text-xs font-medium text-purple-700 mb-2 flex items-center gap-1">
                          <Brain className="w-3 h-3" /> Direct AI Answer
                        </p>
                        <div className="text-sm text-gray-700 markdown-content">
                          <ReactMarkdown>{msg.direct_ai_answer}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {/* Reasoning */}
                    {msg.reasoning && (
                      <p className="text-xs text-gray-400 italic">{msg.reasoning}</p>
                    )}

                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-2">
                      {msg.sources?.length > 0 && (
                        <button
                          onClick={() => toggleSources(msg.id)}
                          className="flex items-center gap-1 text-xs px-2.5 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-600 transition"
                        >
                          <BookOpen className="w-3 h-3" />
                          Sources ({msg.sources.length})
                          <ChevronDown className={`w-3 h-3 transition ${showSources[msg.id] ? 'rotate-180' : ''}`} />
                        </button>
                      )}
                      {msg.ai_summary && (
                        <button
                          onClick={() => toggleComparison(msg.id)}
                          className="flex items-center gap-1 text-xs px-2.5 py-1 bg-purple-50 hover:bg-purple-100 rounded-full text-purple-600 transition"
                        >
                          <Sparkles className="w-3 h-3" />
                          AI vs PDF
                        </button>
                      )}
                      {msg.answer_mode !== 'ai' && (
                        <button
                          onClick={() => fetchRelated(msg)}
                          className="flex items-center gap-1 text-xs px-2.5 py-1 bg-amber-50 hover:bg-amber-100 rounded-full text-amber-700 transition"
                        >
                          {relatedLoading[msg.id] ? <Loader className="w-3 h-3 animate-spin" /> : <Lightbulb className="w-3 h-3" />}
                          Related
                          <ChevronDown className={`w-3 h-3 transition ${showRelated[msg.id] ? 'rotate-180' : ''}`} />
                        </button>
                      )}
                      <button
                        onClick={() => handleFactCheck(msg)}
                        className="flex items-center gap-1 text-xs px-2.5 py-1 bg-emerald-50 hover:bg-emerald-100 rounded-full text-emerald-700 transition"
                        title="Fact-check this answer"
                      >
                        {factCheckLoading[msg.id] ? <Loader className="w-3 h-3 animate-spin" /> : <Shield className="w-3 h-3" />}
                        Verify
                      </button>
                      {msg.answer_mode !== 'ai' && selectedDocs.length > 0 && (
                        <button
                          onClick={() => handleHighlight(msg)}
                          className="flex items-center gap-1 text-xs px-2.5 py-1 bg-cyan-50 hover:bg-cyan-100 rounded-full text-cyan-700 transition"
                          title="Find in PDF"
                        >
                          {highlightLoading[msg.id] ? <Loader className="w-3 h-3 animate-spin" /> : <Highlighter className="w-3 h-3" />}
                          Find in PDF
                        </button>
                      )}
                      {msg.confidence > 0 && (
                        <button
                          onClick={() => handleExplainConfidence(msg)}
                          className="flex items-center gap-1 text-xs px-2.5 py-1 bg-violet-50 hover:bg-violet-100 rounded-full text-violet-700 transition"
                          title="Explain confidence"
                        >
                          <Info className="w-3 h-3" />
                          Why {(msg.confidence * 100).toFixed(0)}%?
                        </button>
                      )}
                      <button
                        onClick={() => handleFeedback(msg.id, true)}
                        className="p-1 text-gray-400 hover:text-green-500 rounded transition"
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleFeedback(msg.id, false)}
                        className="p-1 text-gray-400 hover:text-red-500 rounded transition"
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => speak(msg.id, msg.content)}
                        className={`p-1 rounded transition ${isSpeaking === msg.id ? 'text-primary-500' : 'text-gray-400 hover:text-primary-500'}`}
                        title={isSpeaking === msg.id ? 'Stop speaking' : 'Read aloud'}
                      >
                        {isSpeaking === msg.id ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                      </button>
                      <button
                        onClick={() => copyAnswer(msg.id, msg.content)}
                        className="p-1 text-gray-400 hover:text-gray-600 rounded transition"
                        title="Copy answer"
                      >
                        {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                      <button
                        onClick={() => handleBookmark(msg)}
                        className={`p-1 rounded transition ${bookmarkedIds.has(msg.id) ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                        title={bookmarkedIds.has(msg.id) ? 'Bookmarked' : 'Bookmark this answer'}
                      >
                        {bookmarkedIds.has(msg.id) ? <BookmarkCheck className="w-3.5 h-3.5" /> : <Bookmark className="w-3.5 h-3.5" />}
                      </button>
                      {/* Translate dropdown */}
                      <div className="relative group">
                        <button className="p-1 text-gray-400 hover:text-blue-500 rounded transition" title="Translate">
                          <Globe className="w-3.5 h-3.5" />
                        </button>
                        <div className="absolute bottom-full left-0 mb-1 hidden group-hover:block bg-white border border-gray-200 rounded-lg shadow-lg p-1 z-10 min-w-[100px]">
                          {[['es','Spanish'],['fr','French'],['de','German'],['hi','Hindi'],['zh','Chinese'],['ja','Japanese'],['ko','Korean']].map(([code,name]) => (
                            <button
                              key={code}
                              onClick={() => handleTranslate(msg, code)}
                              className="block w-full text-left px-3 py-1 text-xs text-gray-600 hover:bg-gray-50 rounded"
                            >
                              {translating[`${msg.id}_${code}`] ? '...' : name}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Translation Panel */}
                    {Object.entries(translations).filter(([k]) => k.startsWith(`${msg.id}_`)).map(([key, t]) => (
                      <div key={key} className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                        <p className="text-xs font-medium text-blue-700 mb-1 flex items-center gap-1">
                          <Languages className="w-3 h-3" /> Translation ({t.target_lang})
                        </p>
                        <p className="text-sm text-gray-700">{t.translated}</p>
                      </div>
                    ))}

                    {/* Sources Panel */}
                    {showSources[msg.id] && msg.sources && (
                      <div className="bg-gray-50 rounded-lg p-3 space-y-2 border border-gray-200">
                        <p className="text-xs font-medium text-gray-700">Sources</p>
                        {msg.sources.map((src, i) => (
                          <div key={i} className="flex items-start gap-2 text-xs">
                            <FileText className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
                            <div>
                              <span className="font-medium">{src.file}</span>
                              <span className="text-gray-400"> - Page {src.page}</span>
                              <span className="text-gray-400"> (Score: {(src.score * 100).toFixed(0)}%)</span>
                              {src.content_preview && (
                                <p className="text-gray-500 mt-0.5 line-clamp-2">{src.content_preview}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* AI vs PDF Comparison Panel */}
                    {showComparison[msg.id] && msg.ai_summary && (
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                          <p className="text-xs font-medium text-blue-700 mb-2 flex items-center gap-1">
                            <BookOpen className="w-3 h-3" /> PDF Answer
                          </p>
                          <p className="text-xs text-gray-700">{msg.content}</p>
                        </div>
                        <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                          <p className="text-xs font-medium text-purple-700 mb-2 flex items-center gap-1">
                            <Sparkles className="w-3 h-3" /> AI Summary
                          </p>
                          <p className="text-xs text-gray-700">{msg.ai_summary}</p>
                        </div>
                      </div>
                    )}

                    {/* Suggested Questions */}
                    {msg.suggested_questions?.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {msg.suggested_questions.map((q, i) => (
                          <button
                            key={i}
                            onClick={() => handleSuggestedQuestion(q)}
                            className="text-xs px-3 py-1.5 bg-primary-50 text-primary-700 rounded-full hover:bg-primary-100 transition truncate max-w-xs"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Related Content Panel */}
                    {showRelated[msg.id] && (
                      <div className="bg-amber-50 rounded-lg p-3 space-y-2 border border-amber-200">
                        <p className="text-xs font-medium text-amber-700 flex items-center gap-1">
                          <Lightbulb className="w-3 h-3" /> Related Content from Your Documents
                        </p>
                        {relatedLoading[msg.id] ? (
                          <div className="flex items-center gap-2 py-2">
                            <Loader className="w-3.5 h-3.5 text-amber-500 animate-spin" />
                            <span className="text-xs text-gray-500">Finding related information...</span>
                          </div>
                        ) : (relatedContent[msg.id] || []).length === 0 ? (
                          <p className="text-xs text-gray-500 italic">No additional related content found.</p>
                        ) : (
                          (relatedContent[msg.id] || []).map((item, i) => (
                            <div key={i} className="bg-white rounded-md p-2.5 border border-amber-100 text-xs">
                              <div className="flex items-center gap-2 mb-1">
                                <FileText className="w-3 h-3 text-red-400 shrink-0" />
                                <span className="font-medium text-gray-700">{item.file}</span>
                                <span className="text-gray-400">Page {item.page}</span>
                                <span className="text-gray-300">·</span>
                                <span className="text-amber-600">{(item.score * 100).toFixed(0)}% match</span>
                              </div>
                              <p className="text-gray-600 leading-relaxed">{item.content}</p>
                            </div>
                          ))
                        )}
                      </div>
                    )}

                    {/* Fact-Check Panel */}
                    {showFactCheck[msg.id] && factChecks[msg.id] && (
                      <div className="bg-emerald-50 rounded-lg p-3 space-y-2 border border-emerald-200">
                        <p className="text-xs font-medium text-emerald-700 flex items-center gap-1">
                          <Shield className="w-3 h-3" /> Fact-Check Results
                        </p>
                        {factCheckLoading[msg.id] ? (
                          <div className="flex items-center gap-2 py-2">
                            <Loader className="w-3.5 h-3.5 text-emerald-500 animate-spin" />
                            <span className="text-xs text-gray-500">Verifying claims...</span>
                          </div>
                        ) : (
                          <>
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium text-gray-700">Confidence:</span>
                              <div className="flex-1 bg-emerald-200 rounded-full h-1.5">
                                <div className="bg-emerald-600 h-1.5 rounded-full" style={{ width: `${(factChecks[msg.id].confidence_rating || 0) * 100}%` }} />
                              </div>
                              <span className="text-xs text-emerald-700">{((factChecks[msg.id].confidence_rating || 0) * 100).toFixed(0)}%</span>
                            </div>
                            {factChecks[msg.id].claims?.map((claim, i) => (
                              <div key={i} className="bg-white rounded-md p-2 border border-emerald-100 text-xs">
                                <p className="text-gray-700">{claim}</p>
                              </div>
                            ))}
                            {factChecks[msg.id].verification_notes && (
                              <p className="text-xs text-gray-600 italic">{factChecks[msg.id].verification_notes}</p>
                            )}
                          </>
                        )}
                      </div>
                    )}

                    {/* PDF Highlights Panel */}
                    {showHighlights[msg.id] && highlights[msg.id] && (
                      <div className="bg-cyan-50 rounded-lg p-3 space-y-2 border border-cyan-200">
                        <p className="text-xs font-medium text-cyan-700 flex items-center gap-1">
                          <Highlighter className="w-3 h-3" /> Found in Documents
                        </p>
                        {highlights[msg.id].map((h, i) => (
                          <div key={i} className="bg-white rounded-md p-2.5 border border-cyan-100 text-xs">
                            <div className="flex items-center gap-2 mb-1">
                              <FileText className="w-3 h-3 text-cyan-500 shrink-0" />
                              <span className="font-medium text-gray-700">{h.file}</span>
                              <span className="text-cyan-600">Page {h.page}</span>
                              <span className="text-gray-300">·</span>
                              <span className="text-cyan-500">{(h.score * 100).toFixed(0)}% match</span>
                            </div>
                            <p className="text-gray-600 leading-relaxed bg-yellow-50 p-1.5 rounded border-l-2 border-yellow-400">{h.content}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Confidence Explanation Panel */}
                    {showConfExplain[msg.id] && confidenceExplain[msg.id] && (
                      <div className="bg-violet-50 rounded-lg p-3 space-y-2 border border-violet-200">
                        <p className="text-xs font-medium text-violet-700 flex items-center gap-1">
                          <Info className="w-3 h-3" /> Confidence Breakdown
                        </p>
                        {confidenceExplain[msg.id].factors?.map((f, i) => (
                          <div key={i} className="flex items-center gap-2 text-xs">
                            <span className="text-gray-600 flex-1">{f.factor}</span>
                            <div className="w-20 bg-violet-200 rounded-full h-1.5">
                              <div className="bg-violet-600 h-1.5 rounded-full" style={{ width: `${f.score * 100}%` }} />
                            </div>
                            <span className="text-violet-700 w-8 text-right">{(f.score * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                        {confidenceExplain[msg.id].explanation && (
                          <p className="text-xs text-gray-600 mt-1">{confidenceExplain[msg.id].explanation}</p>
                        )}
                        {confidenceExplain[msg.id].recommendation && (
                          <p className="text-xs text-violet-600 font-medium mt-1">{confidenceExplain[msg.id].recommendation}</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="chat-bubble-assistant flex items-center gap-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm text-gray-500">
                  {answerMode === 'ai' ? 'Generating AI answer...' :
                   answerMode === 'both' ? 'Searching documents & generating AI answer...' :
                   'Searching documents...'}
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 bg-white p-4">
          {/* Answer Mode Selector */}
          <div className="max-w-3xl mx-auto mb-2 flex items-center gap-1">
            <span className="text-xs text-gray-400 mr-1">Mode:</span>
            {[
              { value: 'pdf', label: 'PDF', icon: FileSearch, tip: 'Answer from documents only' },
              { value: 'ai', label: 'AI', icon: Brain, tip: 'Direct AI answer' },
              { value: 'both', label: 'Both', icon: Layers, tip: 'PDF answer + AI answer side by side' },
            ].map(({ value, label, icon: Icon, tip }) => (
              <button
                key={value}
                onClick={() => setAnswerMode(value)}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition ${
                  answerMode === value
                    ? value === 'pdf' ? 'bg-blue-100 text-blue-700'
                      : value === 'ai' ? 'bg-purple-100 text-purple-700'
                      : 'bg-indigo-100 text-indigo-700'
                    : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                }`}
                title={tip}
              >
                <Icon className="w-3 h-3" />
                {label}
              </button>
            ))}
          </div>
          <div className="max-w-3xl mx-auto flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask a question... (Ctrl+/ to focus)"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition"
              disabled={loading}
            />
            <div className="relative">
              <button
                onClick={() => setShowExportMenu(p => !p)}
                className="px-3 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition text-gray-500"
                title="Export chat (Ctrl+E)"
              >
                <Download className="w-5 h-5" />
              </button>
              {showExportMenu && (
                <div className="absolute bottom-full right-0 mb-1 bg-white border border-gray-200 rounded-lg shadow-lg p-1 z-10 min-w-[120px]">
                  {[['md','Markdown'],['txt','Plain Text'],['json','JSON'],['html','HTML']].map(([fmt,label]) => (
                    <button
                      key={fmt}
                      onClick={() => { handleExportFormat(fmt); setShowExportMenu(false); }}
                      className="block w-full text-left px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50 rounded"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => setShowShortcuts((p) => !p)}
              className="px-3 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition text-gray-500"
              title="Keyboard shortcuts (Ctrl+K)"
            >
              <Keyboard className="w-5 h-5" />
            </button>
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="px-5 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Keyboard Shortcuts Modal */}
        {showShortcuts && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowShortcuts(false)}>
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 w-80" onClick={(e) => e.stopPropagation()}>
              <h3 className="font-semibold text-lg mb-4">Keyboard Shortcuts</h3>
              <div className="space-y-3 text-sm">
                {[
                  ['Ctrl + /', 'Focus search input'],
                  ['Ctrl + E', 'Export chat as Markdown'],
                  ['Ctrl + K', 'Show/hide shortcuts'],
                  ['Enter', 'Send message'],
                  ['Escape', 'Close modals / stop TTS'],
                ].map(([key, desc]) => (
                  <div key={key} className="flex justify-between">
                    <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">{key}</kbd>
                    <span className="text-gray-500">{desc}</span>
                  </div>
                ))}
              </div>
              <button onClick={() => setShowShortcuts(false)} className="mt-4 w-full py-2 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm hover:bg-gray-200 dark:hover:bg-gray-600 transition">
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
