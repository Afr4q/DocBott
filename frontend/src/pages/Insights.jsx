/**
 * Insights Page - Smart document analysis with relationship mapping,
 * cross-lingual support, key topics, and document connections.
 */

import React, { useState, useEffect } from 'react';
import { documentsAPI } from '../services/api';
import {
  FileText, Loader, Brain, Globe, Network, Hash,
  ChevronDown, ChevronUp, BookOpen, Sparkles, Languages,
  ArrowRight, BarChart3, Clock, GitCompare, Layers, Award
} from 'lucide-react';
import toast from 'react-hot-toast';

const LANG_NAMES = {
  en: 'English', es: 'Spanish', fr: 'French', de: 'German',
  hi: 'Hindi', ar: 'Arabic', zh: 'Chinese', ja: 'Japanese',
  ko: 'Korean', ru: 'Russian', pt: 'Portuguese', it: 'Italian',
};

export default function Insights() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [insights, setInsights] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [relationships, setRelationships] = useState(null);
  const [relLoading, setRelLoading] = useState(false);
  const [showTopics, setShowTopics] = useState(true);
  // Document Comparison
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  const [comparison, setComparison] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  // Stats Overview
  const [statsOverview, setStatsOverview] = useState(null);

  useEffect(() => {
    documentsAPI.list().then((res) => {
      setDocuments(res.data.filter((d) => d.status === 'processed'));
      setLoading(false);
    }).catch(() => { setLoading(false); toast.error('Failed to load documents'); });

    // Load stats overview
    documentsAPI.statsOverview().then((res) => setStatsOverview(res.data)).catch(() => {});
  }, []);

  const loadInsights = async (doc) => {
    setSelectedDoc(doc);
    setInsightsLoading(true);
    setInsights(null);
    try {
      const res = await documentsAPI.insights(doc.id);
      setInsights(res.data);
    } catch {
      toast.error('Failed to load insights');
    } finally {
      setInsightsLoading(false);
    }
  };

  const loadRelationships = async () => {
    setRelLoading(true);
    try {
      const res = await documentsAPI.relationships();
      setRelationships(res.data);
    } catch {
      toast.error('Failed to analyze relationships');
    } finally {
      setRelLoading(false);
    }
  };

  const runComparison = async () => {
    if (!compareA || !compareB || compareA === compareB) {
      toast.error('Select two different documents to compare');
      return;
    }
    setCompareLoading(true);
    setComparison(null);
    try {
      const res = await documentsAPI.compare({ doc_id_a: parseInt(compareA), doc_id_b: parseInt(compareB) });
      setComparison(res.data);
    } catch {
      toast.error('Failed to compare documents');
    } finally {
      setCompareLoading(false);
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
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Document Insights</h1>
        <p className="text-gray-500 mt-1">Smart analysis, key topics, language detection, and document relationships</p>
      </div>

      {/* Stats Overview Cards */}
      {statsOverview && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-6">
          {[
            { label: 'Documents', value: statsOverview.total_documents, icon: FileText, color: 'blue' },
            { label: 'Processed', value: statsOverview.processed_documents, icon: Award, color: 'green' },
            { label: 'Pages', value: statsOverview.total_pages, icon: BookOpen, color: 'purple' },
            { label: 'Chunks', value: statsOverview.total_chunks, icon: Layers, color: 'indigo' },
            { label: 'Words', value: statsOverview.total_words?.toLocaleString(), icon: Hash, color: 'amber' },
            { label: 'Reading', value: `${statsOverview.estimated_reading_time_min} min`, icon: Clock, color: 'rose' },
            { label: 'FAQs', value: statsOverview.total_faqs, icon: Brain, color: 'teal' },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm text-center">
              <Icon className={`w-5 h-5 mx-auto mb-1 text-${color}-500`} />
              <p className="text-lg font-bold text-gray-900">{value}</p>
              <p className="text-xs text-gray-500">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Document Relationship Map */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Network className="w-5 h-5 text-indigo-500" />
            Document Relationship Map
          </h2>
          <button
            onClick={loadRelationships}
            disabled={relLoading || documents.length < 2}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition disabled:opacity-50 text-sm"
          >
            {relLoading ? <Loader className="w-4 h-4 animate-spin" /> : <Network className="w-4 h-4" />}
            Analyze Relationships
          </button>
        </div>

        {documents.length < 2 && (
          <p className="text-sm text-gray-400 italic">Upload at least 2 documents to see relationships.</p>
        )}

        {relationships && (
          <div className="mt-4">
            {relationships.edges?.length === 0 ? (
              <p className="text-sm text-gray-500 italic">No significant relationships found between documents.</p>
            ) : (
              <div className="space-y-3">
                {/* Nodes */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {relationships.nodes?.map((node) => (
                    <span key={node.id} className="px-3 py-1.5 bg-indigo-50 text-indigo-700 rounded-full text-sm font-medium border border-indigo-200">
                      <FileText className="w-3.5 h-3.5 inline mr-1" />
                      {node.label}
                    </span>
                  ))}
                </div>
                {/* Edges */}
                <div className="space-y-2">
                  {relationships.edges?.map((edge, i) => {
                    const source = relationships.nodes?.find((n) => n.id === edge.source);
                    const target = relationships.nodes?.find((n) => n.id === edge.target);
                    return (
                      <div
                        key={i}
                        className={`flex items-center gap-3 p-3 rounded-lg border ${
                          edge.relationship === 'strong' ? 'bg-green-50 border-green-200' :
                          edge.relationship === 'moderate' ? 'bg-blue-50 border-blue-200' :
                          'bg-gray-50 border-gray-200'
                        }`}
                      >
                        <span className="text-sm font-medium text-gray-700 truncate max-w-[200px]">
                          {source?.label || edge.source}
                        </span>
                        <ArrowRight className="w-4 h-4 text-gray-400 shrink-0" />
                        <span className="text-sm font-medium text-gray-700 truncate max-w-[200px]">
                          {target?.label || edge.target}
                        </span>
                        <span className={`ml-auto px-2 py-0.5 rounded-full text-xs font-medium ${
                          edge.relationship === 'strong' ? 'bg-green-100 text-green-700' :
                          edge.relationship === 'moderate' ? 'bg-blue-100 text-blue-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {(edge.similarity * 100).toFixed(0)}% similar
                        </span>
                        {edge.shared_topics?.length > 0 && (
                          <div className="flex gap-1 ml-2">
                            {edge.shared_topics.slice(0, 3).map((t, j) => (
                              <span key={j} className="px-1.5 py-0.5 bg-white rounded text-xs text-gray-500 border">{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Document Comparison */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2 mb-4">
          <GitCompare className="w-5 h-5 text-orange-500" />
          Document Comparison
        </h2>
        <div className="flex flex-wrap items-end gap-3 mb-4">
          <div className="flex-1 min-w-[180px]">
            <label className="block text-xs font-medium text-gray-500 mb-1">Document A</label>
            <select value={compareA} onChange={(e) => setCompareA(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none">
              <option value="">Select document...</option>
              {documents.map((d) => <option key={d.id} value={d.id}>{d.original_name}</option>)}
            </select>
          </div>
          <div className="flex-1 min-w-[180px]">
            <label className="block text-xs font-medium text-gray-500 mb-1">Document B</label>
            <select value={compareB} onChange={(e) => setCompareB(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none">
              <option value="">Select document...</option>
              {documents.map((d) => <option key={d.id} value={d.id}>{d.original_name}</option>)}
            </select>
          </div>
          <button onClick={runComparison} disabled={compareLoading || !compareA || !compareB} className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition disabled:opacity-50 text-sm">
            {compareLoading ? <Loader className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
            Compare
          </button>
        </div>

        {comparison && (
          <div className="space-y-4">
            {/* Similarity Score */}
            <div className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg border border-orange-200">
              <span className="text-sm font-medium text-gray-700">Overall Similarity:</span>
              <div className="flex-1 bg-orange-200 rounded-full h-2.5">
                <div className="bg-orange-600 h-2.5 rounded-full transition-all" style={{ width: `${comparison.similarity * 100}%` }} />
              </div>
              <span className="text-sm font-bold text-orange-700">{(comparison.similarity * 100).toFixed(1)}%</span>
              <span className="text-xs text-gray-500">({comparison.total_shared_words} shared terms)</span>
            </div>

            {/* Side-by-Side Stats */}
            <div className="grid grid-cols-2 gap-4">
              {[comparison.doc_a, comparison.doc_b].map((doc, idx) => (
                <div key={idx} className={`p-4 rounded-lg border ${idx === 0 ? 'bg-blue-50 border-blue-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  <p className={`text-sm font-semibold ${idx === 0 ? 'text-blue-700' : 'text-emerald-700'} truncate mb-2`}>{doc.name}</p>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                    <span><BookOpen className="w-3 h-3 inline mr-1" />{doc.page_count} pages</span>
                    <span><Hash className="w-3 h-3 inline mr-1" />{doc.word_count?.toLocaleString()} words</span>
                    <span><Clock className="w-3 h-3 inline mr-1" />{doc.reading_time_min} min read</span>
                    <span><Languages className="w-3 h-3 inline mr-1" />{LANG_NAMES[doc.language] || doc.language}</span>
                  </div>
                  {doc.top_unique_words?.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs font-medium text-gray-500 mb-1">Unique topics:</p>
                      <div className="flex flex-wrap gap-1">
                        {doc.top_unique_words.slice(0, 6).map((t) => (
                          <span key={t.word} className={`px-1.5 py-0.5 rounded text-xs ${idx === 0 ? 'bg-blue-100 text-blue-700' : 'bg-emerald-100 text-emerald-700'}`}>{t.word} ({t.count})</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Shared Topics */}
            {comparison.shared_topics?.length > 0 && (
              <div>
                <p className="text-sm font-medium text-gray-700 mb-2">Shared Topics</p>
                <div className="flex flex-wrap gap-2">
                  {comparison.shared_topics.map((t) => (
                    <span key={t.word} className="px-2 py-1 bg-orange-100 text-orange-700 rounded-full text-xs border border-orange-200">
                      {t.word} <span className="text-orange-400">({t.count_a} | {t.count_b})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
            <div className="p-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-900 text-sm">Select Document</h3>
            </div>
            <div className="max-h-[500px] overflow-y-auto">
              {documents.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => loadInsights(doc)}
                  className={`w-full flex items-center gap-3 px-4 py-3 text-left transition border-b border-gray-100 ${
                    selectedDoc?.id === doc.id ? 'bg-primary-50' : 'hover:bg-gray-50'
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

        {/* Insights Panel */}
        <div className="lg:col-span-2">
          {insightsLoading ? (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex items-center justify-center py-20">
              <div className="text-center">
                <Loader className="w-8 h-8 text-primary-500 animate-spin mx-auto mb-3" />
                <p className="text-sm text-gray-500">Analyzing document...</p>
              </div>
            </div>
          ) : insights ? (
            <div className="space-y-4">
              {/* Summary Card */}
              <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2 mb-3">
                  <Sparkles className="w-5 h-5 text-purple-500" />
                  AI Summary
                </h3>
                <p className="text-sm text-gray-700 leading-relaxed">{insights.summary}</p>
              </div>

              {/* Language & Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <Languages className="w-4 h-4 text-blue-500" />
                    <p className="text-xs text-gray-500">Language</p>
                  </div>
                  <p className="text-lg font-bold text-gray-900">
                    {LANG_NAMES[insights.detected_language] || insights.detected_language}
                  </p>
                </div>
                <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <BookOpen className="w-4 h-4 text-green-500" />
                    <p className="text-xs text-gray-500">Pages</p>
                  </div>
                  <p className="text-lg font-bold text-gray-900">{insights.stats.page_count || 0}</p>
                </div>
                <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <Hash className="w-4 h-4 text-indigo-500" />
                    <p className="text-xs text-gray-500">Chunks</p>
                  </div>
                  <p className="text-lg font-bold text-gray-900">{insights.stats.total_chunks}</p>
                </div>
                <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <BarChart3 className="w-4 h-4 text-purple-500" />
                    <p className="text-xs text-gray-500">Est. Words</p>
                  </div>
                  <p className="text-lg font-bold text-gray-900">{insights.stats.estimated_words?.toLocaleString()}</p>
                </div>
              </div>

              {/* Key Topics */}
              <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <button
                  onClick={() => setShowTopics(!showTopics)}
                  className="flex items-center justify-between w-full"
                >
                  <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                    <Brain className="w-5 h-5 text-emerald-500" />
                    Key Topics
                  </h3>
                  {showTopics ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                </button>
                {showTopics && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {insights.key_topics?.map((topic, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm border"
                        style={{
                          backgroundColor: `hsl(${(i * 36) % 360}, 85%, 95%)`,
                          borderColor: `hsl(${(i * 36) % 360}, 60%, 80%)`,
                          color: `hsl(${(i * 36) % 360}, 50%, 35%)`,
                        }}
                      >
                        {topic.word}
                        <span className="text-xs opacity-70">({topic.count})</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Source Types */}
              {insights.stats.source_types?.length > 0 && (
                <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                  <h3 className="font-semibold text-gray-900 mb-3">Content Sources</h3>
                  <div className="flex gap-3">
                    {insights.stats.source_types.map((type) => (
                      <span key={type} className="px-3 py-1.5 bg-gray-50 rounded-full text-sm text-gray-600 border capitalize">
                        {type}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex items-center justify-center py-20">
              <div className="text-center">
                <Brain className="w-16 h-16 text-gray-200 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-600">Select a document</h3>
                <p className="text-sm text-gray-400 mt-1">Choose a document to view its AI-generated insights</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
