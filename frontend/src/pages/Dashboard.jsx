/**
 * Dashboard Page - Overview of documents and system status.
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { documentsAPI, tagsAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import {
  FileText, Upload, MessageSquare, Trash2, RefreshCw,
  CheckCircle, Clock, AlertCircle, Loader,
  BarChart3, Tag, X, Plus, BookOpen, Hash
} from 'lucide-react';
import toast from 'react-hot-toast';

const statusConfig = {
  uploaded: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-50', label: 'Uploaded' },
  processing: { icon: Loader, color: 'text-blue-500', bg: 'bg-blue-50', label: 'Processing' },
  processed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50', label: 'Ready' },
  failed: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-50', label: 'Failed' },
};

export default function Dashboard() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [docTags, setDocTags] = useState({}); // { docId: [{ id, tag }] }
  const [tagInput, setTagInput] = useState({}); // { docId: 'inputValue' }
  const { user } = useAuth();

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const res = await documentsAPI.list();
      setDocuments(res.data);
      // Load tags for each document
      const tagMap = {};
      await Promise.all(
        res.data.map(async (doc) => {
          try {
            const tagRes = await tagsAPI.list(doc.id);
            tagMap[doc.id] = tagRes.data;
          } catch { tagMap[doc.id] = []; }
        })
      );
      setDocTags(tagMap);
    } catch (err) {
      toast.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await documentsAPI.analytics();
      setAnalytics(res.data);
    } catch { /* analytics is optional */ }
  };

  useEffect(() => { fetchDocuments(); fetchAnalytics(); }, []);

  const handleDelete = async (id, name) => {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
      await documentsAPI.delete(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      toast.success('Document deleted');
    } catch (err) {
      toast.error('Failed to delete document');
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return '-';
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    });
  };

  const addTag = async (docId) => {
    const tagValue = (tagInput[docId] || '').trim();
    if (!tagValue) return;
    try {
      await tagsAPI.add(docId, tagValue);
      const res = await tagsAPI.list(docId);
      setDocTags((prev) => ({ ...prev, [docId]: res.data }));
      setTagInput((prev) => ({ ...prev, [docId]: '' }));
      toast.success('Tag added');
    } catch {
      toast.error('Failed to add tag');
    }
  };

  const removeTag = async (docId, tagId) => {
    try {
      await tagsAPI.remove(docId, tagId);
      setDocTags((prev) => ({
        ...prev,
        [docId]: (prev[docId] || []).filter((t) => t.id !== tagId),
      }));
    } catch {
      toast.error('Failed to remove tag');
    }
  };

  const stats = {
    total: documents.length,
    processed: documents.filter((d) => d.status === 'processed').length,
    processing: documents.filter((d) => d.status === 'processing').length,
    failed: documents.filter((d) => d.status === 'failed').length,
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Welcome, {user?.username}</h1>
          <p className="text-gray-500 mt-1">Manage your documents and start asking questions</p>
        </div>
        <div className="flex gap-3">
          <button onClick={fetchDocuments} className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition text-sm">
            <RefreshCw className="w-4 h-4" /> Refresh
          </button>
          <Link to="/upload" className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition text-sm">
            <Upload className="w-4 h-4" /> Upload PDF
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Total Documents', value: stats.total, color: 'bg-blue-500' },
          { label: 'Ready', value: stats.processed, color: 'bg-green-500' },
          { label: 'Processing', value: stats.processing, color: 'bg-yellow-500' },
          { label: 'Failed', value: stats.failed, color: 'bg-red-500' },
        ].map((stat) => (
          <div key={stat.label} className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${stat.color}`} />
              <p className="text-sm text-gray-500">{stat.label}</p>
            </div>
            <p className="text-3xl font-bold text-gray-900 mt-2">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        <Link
          to="/chat"
          className="flex items-center gap-4 p-6 bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl text-white hover:from-primary-600 hover:to-primary-700 transition shadow-lg"
        >
          <MessageSquare className="w-8 h-8" />
          <div>
            <p className="font-semibold text-lg">Ask Questions</p>
            <p className="text-primary-100 text-sm">Chat with your documents using AI</p>
          </div>
        </Link>
        <Link
          to="/summary"
          className="flex items-center gap-4 p-6 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl text-white hover:from-purple-600 hover:to-purple-700 transition shadow-lg"
        >
          <FileText className="w-8 h-8" />
          <div>
            <p className="font-semibold text-lg">Compare & Summarize</p>
            <p className="text-purple-100 text-sm">Compare answers across documents</p>
          </div>
        </Link>
      </div>

      {/* Analytics Section */}
      {analytics && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary-500" /> Document Analytics
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Pages', value: analytics.total_pages, icon: BookOpen, color: 'text-blue-500' },
              { label: 'Text Chunks', value: analytics.total_chunks, icon: Hash, color: 'text-indigo-500' },
              { label: 'Est. Words', value: analytics.estimated_words?.toLocaleString(), icon: FileText, color: 'text-purple-500' },
              { label: 'Total Queries', value: analytics.total_queries, icon: MessageSquare, color: 'text-green-500' },
            ].map((item) => (
              <div key={item.label} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <item.icon className={`w-4 h-4 ${item.color}`} />
                  <p className="text-xs text-gray-500">{item.label}</p>
                </div>
                <p className="text-2xl font-bold text-gray-900">{item.value || 0}</p>
              </div>
            ))}
          </div>
          {/* Source breakdown */}
          {Object.keys(analytics.source_breakdown || {}).length > 0 && (
            <div className="mt-4 bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
              <p className="text-sm font-medium text-gray-700 mb-3">Content Sources</p>
              <div className="flex flex-wrap gap-3">
                {Object.entries(analytics.source_breakdown).map(([type, count]) => (
                  <div key={type} className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-full text-sm">
                    <div className={`w-2 h-2 rounded-full ${
                      type === 'text' ? 'bg-blue-500' : type === 'table' ? 'bg-green-500' : 'bg-orange-500'
                    }`} />
                    <span className="text-gray-600 capitalize">{type}</span>
                    <span className="font-semibold text-gray-900">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Documents Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Your Documents</h2>
        </div>

        {loading ? (
          <div className="flex justify-center py-12">
            <Loader className="w-8 h-8 text-primary-500 animate-spin" />
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No documents uploaded yet</p>
            <Link to="/upload" className="text-primary-600 hover:underline text-sm mt-1 inline-block">
              Upload your first PDF
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider">
                <tr>
                  <th className="px-6 py-3 text-left">Document</th>
                  <th className="px-6 py-3 text-left">Status</th>
                  <th className="px-6 py-3 text-left">Tags</th>
                  <th className="px-6 py-3 text-left">Pages</th>
                  <th className="px-6 py-3 text-left">Size</th>
                  <th className="px-6 py-3 text-left">Uploaded</th>
                  <th className="px-6 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {documents.map((doc) => {
                  const status = statusConfig[doc.status] || statusConfig.uploaded;
                  const StatusIcon = status.icon;
                  const tags = docTags[doc.id] || [];
                  return (
                    <tr key={doc.id} className="hover:bg-gray-50 transition">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <FileText className="w-5 h-5 text-red-500 shrink-0" />
                          <span className="text-sm font-medium text-gray-900 truncate max-w-xs">
                            {doc.original_name}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${status.bg} ${status.color}`}>
                          <StatusIcon className={`w-3.5 h-3.5 ${doc.status === 'processing' ? 'animate-spin' : ''}`} />
                          {status.label}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-wrap items-center gap-1 max-w-xs">
                          {tags.map((t) => (
                            <span key={t.id} className="inline-flex items-center gap-0.5 px-2 py-0.5 bg-primary-50 text-primary-700 rounded-full text-xs">
                              <Tag className="w-2.5 h-2.5" />
                              {t.tag}
                              <button onClick={() => removeTag(doc.id, t.id)} className="ml-0.5 hover:text-red-500">
                                <X className="w-2.5 h-2.5" />
                              </button>
                            </span>
                          ))}
                          <div className="inline-flex items-center">
                            <input
                              type="text"
                              value={tagInput[doc.id] || ''}
                              onChange={(e) => setTagInput((prev) => ({ ...prev, [doc.id]: e.target.value }))}
                              onKeyDown={(e) => e.key === 'Enter' && addTag(doc.id)}
                              placeholder="+ tag"
                              className="w-16 px-1.5 py-0.5 text-xs border border-gray-200 rounded focus:ring-1 focus:ring-primary-500 outline-none"
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">{doc.page_count || '-'}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{formatSize(doc.file_size)}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{formatDate(doc.created_at)}</td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => handleDelete(doc.id, doc.original_name)}
                          className="p-1.5 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50 transition"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
