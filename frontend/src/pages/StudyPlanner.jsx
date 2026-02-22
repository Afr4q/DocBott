/**
 * StudyPlanner - A simple study planner to organize tasks and track progress.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  CalendarDays, Plus, Trash2, Edit3, CheckCircle2, Circle,
  Clock, AlertTriangle, Filter, BarChart3, FileText, ArrowUpDown,
  X, Save, PlayCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { plannerAPI, documentsAPI } from '../services/api';

const PRIORITY_CONFIG = {
  high: { label: 'High', color: 'text-red-600', bg: 'bg-red-50 border-red-200', badge: 'bg-red-100 text-red-700' },
  medium: { label: 'Medium', color: 'text-amber-600', bg: 'bg-amber-50 border-amber-200', badge: 'bg-amber-100 text-amber-700' },
  low: { label: 'Low', color: 'text-green-600', bg: 'bg-green-50 border-green-200', badge: 'bg-green-100 text-green-700' },
};

const STATUS_CONFIG = {
  todo: { label: 'To Do', icon: Circle, color: 'text-gray-500', bg: 'bg-gray-100' },
  in_progress: { label: 'In Progress', icon: PlayCircle, color: 'text-blue-500', bg: 'bg-blue-100' },
  done: { label: 'Done', icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-100' },
};

export default function StudyPlanner() {
  const [tasks, setTasks] = useState([]);
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');

  // Form state
  const [form, setForm] = useState({
    title: '',
    description: '',
    document_id: '',
    due_date: '',
    priority: 'medium',
  });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [tasksRes, statsRes, docsRes] = await Promise.all([
        plannerAPI.getTasks(),
        plannerAPI.stats(),
        documentsAPI.list(),
      ]);
      setTasks(tasksRes.data);
      setStats(statsRes.data);
      setDocuments(docsRes.data || []);
    } catch (err) {
      toast.error('Failed to load planner data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const resetForm = () => {
    setForm({ title: '', description: '', document_id: '', due_date: '', priority: 'medium' });
    setEditingTask(null);
    setShowForm(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.title.trim()) {
      toast.error('Task title is required');
      return;
    }
    try {
      const payload = {
        title: form.title.trim(),
        description: form.description.trim(),
        document_id: form.document_id ? parseInt(form.document_id) : null,
        due_date: form.due_date || null,
        priority: form.priority,
      };

      if (editingTask) {
        await plannerAPI.updateTask(editingTask.id, payload);
        toast.success('Task updated');
      } else {
        await plannerAPI.createTask(payload);
        toast.success('Task created');
      }
      resetForm();
      loadData();
    } catch (err) {
      toast.error(editingTask ? 'Failed to update task' : 'Failed to create task');
    }
  };

  const handleStatusChange = async (task, newStatus) => {
    try {
      await plannerAPI.updateTask(task.id, { status: newStatus });
      loadData();
      if (newStatus === 'done') toast.success('Task completed! 🎉');
    } catch {
      toast.error('Failed to update status');
    }
  };

  const handleDelete = async (taskId) => {
    if (!window.confirm('Delete this task?')) return;
    try {
      await plannerAPI.deleteTask(taskId);
      toast.success('Task deleted');
      loadData();
    } catch {
      toast.error('Failed to delete task');
    }
  };

  const handleEdit = (task) => {
    setForm({
      title: task.title,
      description: task.description || '',
      document_id: task.document_id || '',
      due_date: task.due_date ? task.due_date.slice(0, 16) : '',
      priority: task.priority,
    });
    setEditingTask(task);
    setShowForm(true);
  };

  const isOverdue = (task) => {
    if (!task.due_date || task.status === 'done') return false;
    return new Date(task.due_date) < new Date();
  };

  const filteredTasks = tasks.filter((t) => {
    if (filterStatus !== 'all' && t.status !== filterStatus) return false;
    if (filterPriority !== 'all' && t.priority !== filterPriority) return false;
    return true;
  });

  const groupedTasks = {
    todo: filteredTasks.filter(t => t.status === 'todo'),
    in_progress: filteredTasks.filter(t => t.status === 'in_progress'),
    done: filteredTasks.filter(t => t.status === 'done'),
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto" />
          <p className="mt-4 text-gray-500">Loading study planner...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
              <div className="w-10 h-10 bg-indigo-100 rounded-xl flex items-center justify-center">
                <CalendarDays className="w-6 h-6 text-indigo-600" />
              </div>
              Study Planner
            </h1>
            <p className="text-gray-500 mt-1">Organize your study tasks and track your progress</p>
          </div>
          <button
            onClick={() => { resetForm(); setShowForm(true); }}
            className="flex items-center gap-2 px-4 py-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition font-medium"
          >
            <Plus className="w-5 h-5" />
            New Task
          </button>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: 'Total', value: stats.total, icon: BarChart3, color: 'text-gray-600', bg: 'bg-gray-100' },
              { label: 'To Do', value: stats.todo, icon: Circle, color: 'text-gray-500', bg: 'bg-gray-50' },
              { label: 'In Progress', value: stats.in_progress, icon: PlayCircle, color: 'text-blue-600', bg: 'bg-blue-50' },
              { label: 'Completed', value: stats.done, icon: CheckCircle2, color: 'text-green-600', bg: 'bg-green-50' },
              { label: 'Overdue', value: stats.overdue, icon: AlertTriangle, color: 'text-red-600', bg: 'bg-red-50' },
            ].map(({ label, value, icon: Icon, color, bg }) => (
              <div key={label} className={`${bg} rounded-xl p-4 border`}>
                <div className="flex items-center gap-2 mb-1">
                  <Icon className={`w-4 h-4 ${color}`} />
                  <span className="text-xs text-gray-500 font-medium">{label}</span>
                </div>
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
              </div>
            ))}
            {stats.total > 0 && (
              <div className="col-span-2 md:col-span-5 bg-white rounded-xl p-4 border">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-500">Overall Progress</span>
                  <span className="text-sm font-semibold text-indigo-600">{stats.completion_rate}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500"
                    style={{ width: `${stats.completion_rate}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Filters */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-500">Filters:</span>
          </div>
          <div className="flex items-center gap-1">
            {['all', 'todo', 'in_progress', 'done'].map((s) => (
              <button
                key={s}
                onClick={() => setFilterStatus(s)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium transition ${
                  filterStatus === s
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {s === 'all' ? 'All' : STATUS_CONFIG[s]?.label || s}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            {['all', 'high', 'medium', 'low'].map((p) => (
              <button
                key={p}
                onClick={() => setFilterPriority(p)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium transition ${
                  filterPriority === p
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {p === 'all' ? 'All Priorities' : PRIORITY_CONFIG[p]?.label || p}
              </button>
            ))}
          </div>
        </div>

        {/* Task Board */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {Object.entries(groupedTasks).map(([status, statusTasks]) => {
            const config = STATUS_CONFIG[status];
            const StatusIcon = config.icon;
            return (
              <div key={status} className="space-y-3">
                <div className="flex items-center gap-2 px-1">
                  <StatusIcon className={`w-5 h-5 ${config.color}`} />
                  <h3 className="font-semibold text-gray-700">{config.label}</h3>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${config.bg} ${config.color} font-medium`}>
                    {statusTasks.length}
                  </span>
                </div>

                <div className="space-y-3 min-h-[100px]">
                  {statusTasks.length === 0 ? (
                    <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 text-center text-gray-400 text-sm">
                      No tasks
                    </div>
                  ) : (
                    statusTasks.map((task) => {
                      const pConfig = PRIORITY_CONFIG[task.priority] || PRIORITY_CONFIG.medium;
                      const overdue = isOverdue(task);
                      return (
                        <div
                          key={task.id}
                          className={`bg-white rounded-xl border p-4 shadow-sm hover:shadow-md transition ${
                            overdue ? 'border-red-300 bg-red-50/30' : 'border-gray-200'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-2 mb-2">
                            <h4 className={`font-medium text-sm ${task.status === 'done' ? 'line-through text-gray-400' : 'text-gray-800'}`}>
                              {task.title}
                            </h4>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${pConfig.badge}`}>
                              {pConfig.label}
                            </span>
                          </div>

                          {task.description && (
                            <p className="text-xs text-gray-500 mb-2 line-clamp-2">{task.description}</p>
                          )}

                          {task.document_name && (
                            <div className="flex items-center gap-1 text-xs text-indigo-600 mb-2">
                              <FileText className="w-3 h-3" />
                              <span className="truncate">{task.document_name}</span>
                            </div>
                          )}

                          {task.due_date && (
                            <div className={`flex items-center gap-1 text-xs mb-3 ${overdue ? 'text-red-600 font-medium' : 'text-gray-400'}`}>
                              {overdue ? <AlertTriangle className="w-3 h-3" /> : <Clock className="w-3 h-3" />}
                              <span>
                                {overdue ? 'Overdue: ' : 'Due: '}
                                {new Date(task.due_date).toLocaleDateString('en-US', {
                                  month: 'short', day: 'numeric', year: 'numeric',
                                  hour: '2-digit', minute: '2-digit',
                                })}
                              </span>
                            </div>
                          )}

                          {/* Actions */}
                          <div className="flex items-center gap-1 pt-2 border-t border-gray-100">
                            {status !== 'done' && (
                              <button
                                onClick={() => handleStatusChange(task, status === 'todo' ? 'in_progress' : 'done')}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg bg-gray-50 hover:bg-indigo-50 text-gray-600 hover:text-indigo-600 transition"
                                title={status === 'todo' ? 'Start' : 'Complete'}
                              >
                                {status === 'todo' ? <PlayCircle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                                {status === 'todo' ? 'Start' : 'Done'}
                              </button>
                            )}
                            {status === 'done' && (
                              <button
                                onClick={() => handleStatusChange(task, 'todo')}
                                className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg bg-gray-50 hover:bg-amber-50 text-gray-600 hover:text-amber-600 transition"
                              >
                                <Circle className="w-3 h-3" /> Reopen
                              </button>
                            )}
                            <button
                              onClick={() => handleEdit(task)}
                              className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg bg-gray-50 hover:bg-blue-50 text-gray-600 hover:text-blue-600 transition ml-auto"
                            >
                              <Edit3 className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => handleDelete(task.id)}
                              className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg bg-gray-50 hover:bg-red-50 text-gray-600 hover:text-red-600 transition"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Empty State */}
        {tasks.length === 0 && (
          <div className="text-center py-16 bg-white rounded-2xl border border-gray-200">
            <CalendarDays className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">No study tasks yet</h3>
            <p className="text-gray-500 mb-6">Create your first task to start organizing your study sessions</p>
            <button
              onClick={() => { resetForm(); setShowForm(true); }}
              className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition font-medium"
            >
              Create First Task
            </button>
          </div>
        )}

        {/* Task Form Modal */}
        {showForm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => resetForm()}>
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between p-5 border-b">
                <h2 className="text-lg font-semibold text-gray-800">
                  {editingTask ? 'Edit Task' : 'New Study Task'}
                </h2>
                <button onClick={resetForm} className="p-1.5 rounded-lg hover:bg-gray-100 transition">
                  <X className="w-5 h-5 text-gray-400" />
                </button>
              </div>

              <form onSubmit={handleSubmit} className="p-5 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Title *</label>
                  <input
                    type="text"
                    value={form.title}
                    onChange={(e) => setForm({ ...form, title: e.target.value })}
                    className="w-full px-3 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none"
                    placeholder="e.g., Read Chapter 5 - Data Structures"
                    autoFocus
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                  <textarea
                    value={form.description}
                    onChange={(e) => setForm({ ...form, description: e.target.value })}
                    className="w-full px-3 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none resize-none"
                    rows={3}
                    placeholder="Add notes or details..."
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Due Date</label>
                    <input
                      type="datetime-local"
                      value={form.due_date}
                      onChange={(e) => setForm({ ...form, due_date: e.target.value })}
                      className="w-full px-3 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                    <select
                      value={form.priority}
                      onChange={(e) => setForm({ ...form, priority: e.target.value })}
                      className="w-full px-3 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-sm"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Link to Document (optional)</label>
                  <select
                    value={form.document_id}
                    onChange={(e) => setForm({ ...form, document_id: e.target.value })}
                    className="w-full px-3 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-sm"
                  >
                    <option value="">No document</option>
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>{doc.original_name}</option>
                    ))}
                  </select>
                </div>

                <div className="flex justify-end gap-3 pt-2">
                  <button
                    type="button"
                    onClick={resetForm}
                    className="px-4 py-2.5 text-sm text-gray-600 bg-gray-100 rounded-xl hover:bg-gray-200 transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex items-center gap-2 px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition font-medium"
                  >
                    <Save className="w-4 h-4" />
                    {editingTask ? 'Update Task' : 'Create Task'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
