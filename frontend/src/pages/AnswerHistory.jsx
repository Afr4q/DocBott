/**
 * Answer History Page – Browse past chat sessions, view messages,
 * search through Q&A history, and manage bookmarks.
 */

import React, { useState, useEffect } from 'react';
import { chatAPI, bookmarksAPI } from '../services/api';
import ReactMarkdown from 'react-markdown';
import {
  MessageSquare, ChevronRight, Search, Trash2,
  Clock, Bookmark, BookmarkCheck, Loader, X,
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function AnswerHistory() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [bookmarks, setBookmarks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msgLoading, setMsgLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState('sessions'); // 'sessions' | 'bookmarks'

  // Load sessions + bookmarks on mount
  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const [sessRes, bmRes] = await Promise.all([
          chatAPI.sessions(),
          bookmarksAPI.list(),
        ]);
        setSessions(sessRes.data.sessions || sessRes.data || []);
        setBookmarks(bmRes.data || []);
      } catch {
        toast.error('Failed to load history');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const openSession = async (session) => {
    try {
      setMsgLoading(true);
      setSelectedSession(session);
      const res = await chatAPI.history(session.id);
      setMessages(res.data.messages || res.data || []);
    } catch {
      toast.error('Failed to load messages');
    } finally {
      setMsgLoading(false);
    }
  };

  const deleteSession = async (id) => {
    if (!confirm('Delete this conversation?')) return;
    try {
      await chatAPI.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (selectedSession?.id === id) {
        setSelectedSession(null);
        setMessages([]);
      }
      toast.success('Session deleted');
    } catch {
      toast.error('Failed to delete session');
    }
  };

  const deleteBookmark = async (id) => {
    try {
      await bookmarksAPI.delete(id);
      setBookmarks((prev) => prev.filter((b) => b.id !== id));
      toast.success('Bookmark removed');
    } catch {
      toast.error('Failed to remove bookmark');
    }
  };

  // Filter helpers
  const filteredSessions = sessions.filter((s) =>
    (s.title || `Session ${s.id}`).toLowerCase().includes(search.toLowerCase())
  );
  const filteredBookmarks = bookmarks.filter(
    (b) =>
      b.query.toLowerCase().includes(search.toLowerCase()) ||
      b.answer.toLowerCase().includes(search.toLowerCase())
  );

  const formatDate = (d) =>
    d ? new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <Loader className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Left Panel – Sessions / Bookmarks list */}
      <div className="w-80 border-r border-gray-200 bg-white flex flex-col">
        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setTab('sessions')}
            className={`flex-1 py-3 text-sm font-medium transition ${
              tab === 'sessions'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <MessageSquare className="w-4 h-4 inline mr-1.5" />
            Sessions ({sessions.length})
          </button>
          <button
            onClick={() => setTab('bookmarks')}
            className={`flex-1 py-3 text-sm font-medium transition ${
              tab === 'bookmarks'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Bookmark className="w-4 h-4 inline mr-1.5" />
            Bookmarks ({bookmarks.length})
          </button>
        </div>

        {/* Search */}
        <div className="p-3">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={tab === 'sessions' ? 'Search sessions...' : 'Search bookmarks...'}
              className="w-full pl-9 pr-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
            />
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {tab === 'sessions' ? (
            filteredSessions.length === 0 ? (
              <p className="text-sm text-gray-400 text-center mt-8">No sessions found</p>
            ) : (
              filteredSessions.map((s) => (
                <div
                  key={s.id}
                  className={`flex items-center gap-2 px-4 py-3 cursor-pointer border-b border-gray-100 transition ${
                    selectedSession?.id === s.id ? 'bg-primary-50' : 'hover:bg-gray-50'
                  }`}
                  onClick={() => openSession(s)}
                >
                  <MessageSquare className="w-4 h-4 text-gray-400 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {s.title || `Session ${s.id}`}
                    </p>
                    <p className="text-xs text-gray-400 flex items-center gap-1 mt-0.5">
                      <Clock className="w-3 h-3" /> {formatDate(s.created_at)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}
                    className="p-1 text-gray-300 hover:text-red-500 rounded transition"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                  <ChevronRight className="w-4 h-4 text-gray-300" />
                </div>
              ))
            )
          ) : filteredBookmarks.length === 0 ? (
            <p className="text-sm text-gray-400 text-center mt-8">No bookmarks found</p>
          ) : (
            filteredBookmarks.map((b) => (
              <div
                key={b.id}
                className="px-4 py-3 border-b border-gray-100 hover:bg-gray-50 transition"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900 truncate">{b.query}</p>
                    <p className="text-xs text-gray-500 line-clamp-2 mt-1">{b.answer}</p>
                    <div className="flex items-center gap-2 mt-1.5">
                      {b.confidence != null && (
                        <span className="text-xs text-gray-400">
                          {(b.confidence * 100).toFixed(0)}% confidence
                        </span>
                      )}
                      <span className="text-xs text-gray-300">{formatDate(b.created_at)}</span>
                    </div>
                    {b.note && (
                      <p className="text-xs text-primary-600 mt-1 italic">📝 {b.note}</p>
                    )}
                  </div>
                  <button
                    onClick={() => deleteBookmark(b.id)}
                    className="p-1 text-gray-300 hover:text-red-500 rounded transition shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right Panel – Message Detail */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {selectedSession && tab === 'sessions' ? (
          <>
            <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <div>
                <h2 className="font-semibold text-gray-900">
                  {selectedSession.title || `Session ${selectedSession.id}`}
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  {formatDate(selectedSession.created_at)} · {messages.length} messages
                </p>
              </div>
              <button
                onClick={() => { setSelectedSession(null); setMessages([]); }}
                className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {msgLoading ? (
                <div className="flex justify-center py-12">
                  <Loader className="w-6 h-6 text-primary-500 animate-spin" />
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-2xl px-4 py-3 rounded-2xl text-sm ${
                      msg.role === 'user'
                        ? 'bg-primary-600 text-white rounded-br-md'
                        : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'
                    }`}>
                      {msg.role === 'user' ? (
                        <p>{msg.content}</p>
                      ) : (
                        <>
                          <div className="markdown-content">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                          </div>
                          {(msg.confidence || msg.sources) && (
                            <div className="mt-2 pt-2 border-t border-gray-100 space-y-1">
                              {msg.confidence > 0 && (
                                <div className="flex items-center gap-2 text-xs text-gray-400">
                                  <div className={`w-2 h-2 rounded-full ${msg.confidence > 0.7 ? 'bg-green-500' : msg.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`} />
                                  Confidence: {(msg.confidence * 100).toFixed(0)}%
                                </div>
                              )}
                              {msg.sources?.length > 0 && (
                                <div className="text-xs text-gray-400">
                                  Sources: {msg.sources.map((s, j) => (
                                    <span key={j}>{s.file || 'Document'} (p.{s.page}){j < msg.sources.length - 1 ? ', ' : ''}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center px-6">
            {tab === 'sessions' ? (
              <>
                <MessageSquare className="w-16 h-16 text-gray-200 mb-4" />
                <h3 className="text-lg font-medium text-gray-600">Select a session</h3>
                <p className="text-sm text-gray-400 mt-1">Choose a conversation from the left to view its messages</p>
              </>
            ) : (
              <>
                <BookmarkCheck className="w-16 h-16 text-gray-200 mb-4" />
                <h3 className="text-lg font-medium text-gray-600">Your Bookmarks</h3>
                <p className="text-sm text-gray-400 mt-1">Saved Q&A pairs appear in the left panel. Bookmark answers from the chat page!</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
