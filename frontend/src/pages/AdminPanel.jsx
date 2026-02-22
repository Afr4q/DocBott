/**
 * Admin Panel - System administration for admin users.
 * User management, role assignment, system stats, activity monitoring.
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { adminAPI } from '../services/api';
import {
  Users, Shield, FileText, MessageSquare, BarChart3, Loader,
  Trash2, UserCheck, UserX, ChevronDown, Activity, Clock,
  Database, Brain, Star, AlertTriangle, TrendingUp
} from 'lucide-react';
import toast from 'react-hot-toast';

const ROLE_COLORS = {
  admin: 'bg-red-100 text-red-700 border-red-200',
  teacher: 'bg-blue-100 text-blue-700 border-blue-200',
  researcher: 'bg-purple-100 text-purple-700 border-purple-200',
  student: 'bg-green-100 text-green-700 border-green-200',
};

export default function AdminPanel() {
  const { user } = useAuth();
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [editingRole, setEditingRole] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [usersRes, statsRes] = await Promise.all([
        adminAPI.users(),
        adminAPI.stats(),
      ]);
      setUsers(usersRes.data);
      setStats(statsRes.data);
    } catch (err) {
      if (err.response?.status === 403) {
        toast.error('Access denied. Admin role required.');
      } else {
        toast.error('Failed to load admin data');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRoleChange = async (userId, newRole) => {
    try {
      await adminAPI.updateRole(userId, newRole);
      toast.success('Role updated');
      setEditingRole(null);
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to update role');
    }
  };

  const handleToggleActive = async (userId) => {
    try {
      await adminAPI.toggleActive(userId);
      toast.success('User status updated');
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to toggle user');
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (!window.confirm(`Are you sure you want to delete user "${username}" and all their data?`)) return;
    try {
      await adminAPI.deleteUser(userId);
      toast.success(`User ${username} deleted`);
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to delete user');
    }
  };

  if (user?.role !== 'admin') {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-3" />
          <h2 className="text-xl font-bold text-red-700">Access Denied</h2>
          <p className="text-red-500 mt-2">This page is only accessible to administrators.</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-4rem)]">
        <Loader className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Shield className="w-7 h-7 text-red-500" />
          Admin Panel
        </h1>
        <p className="text-gray-500 mt-1">System administration and user management</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 mb-6 border-b border-gray-200">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'users', label: 'Users', icon: Users },
          { id: 'activity', label: 'Activity', icon: Activity },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition ${
              activeTab === id
                ? 'border-primary-600 text-primary-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && stats && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Users', value: stats.users.total, icon: Users, color: 'blue', sub: `${stats.users.active} active` },
              { label: 'Documents', value: stats.documents.total, icon: FileText, color: 'green', sub: `${stats.documents.processed} processed` },
              { label: 'Chat Messages', value: stats.chat.total_messages, icon: MessageSquare, color: 'purple', sub: `${stats.chat.total_sessions} sessions` },
              { label: 'Total FAQs', value: stats.content.total_faqs, icon: Brain, color: 'amber', sub: `${stats.content.total_chunks} chunks` },
            ].map(({ label, value, icon: Icon, color, sub }) => (
              <div key={label} className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`w-5 h-5 text-${color}-500`} />
                  <span className="text-xs text-gray-400">{sub}</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">{value?.toLocaleString()}</p>
                <p className="text-xs text-gray-500 mt-1">{label}</p>
              </div>
            ))}
          </div>

          {/* Role Distribution */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4 text-gray-400" />
              User Role Distribution
            </h3>
            <div className="grid grid-cols-4 gap-3">
              {Object.entries(stats.users.roles || {}).map(([role, count]) => (
                <div key={role} className={`px-4 py-3 rounded-lg border text-center ${ROLE_COLORS[role] || 'bg-gray-50'}`}>
                  <p className="text-xl font-bold">{count}</p>
                  <p className="text-xs font-medium capitalize">{role}s</p>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              Last 7 Days
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <p className="text-lg font-bold text-blue-700">{stats.recent.new_users}</p>
                <p className="text-xs text-blue-500">New Users</p>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <p className="text-lg font-bold text-green-700">{stats.recent.new_documents}</p>
                <p className="text-xs text-green-500">New Documents</p>
              </div>
              <div className="text-center p-3 bg-purple-50 rounded-lg">
                <p className="text-lg font-bold text-purple-700">{stats.recent.new_messages}</p>
                <p className="text-xs text-purple-500">Chat Messages</p>
              </div>
            </div>
          </div>

          {/* Document Health */}
          {stats.documents.failed > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-700">{stats.documents.failed} documents failed processing</p>
                <p className="text-xs text-red-500">These documents may need to be re-uploaded.</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">User</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Email</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Role</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Status</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Docs</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Joined</th>
                  <th className="text-right px-4 py-3 font-medium text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {users.map((u) => (
                  <tr key={u.id} className="hover:bg-gray-50 transition">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold">
                          {u.username.charAt(0).toUpperCase()}
                        </div>
                        <span className="font-medium text-gray-900">{u.username}</span>
                        {u.id === user.id && (
                          <span className="text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">You</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-600">{u.email}</td>
                    <td className="px-4 py-3">
                      {editingRole === u.id ? (
                        <select
                          defaultValue={u.role}
                          onChange={(e) => handleRoleChange(u.id, e.target.value)}
                          onBlur={() => setEditingRole(null)}
                          autoFocus
                          className="px-2 py-1 border border-gray-300 rounded text-xs focus:ring-2 focus:ring-primary-500 outline-none"
                        >
                          <option value="student">Student</option>
                          <option value="teacher">Teacher</option>
                          <option value="researcher">Researcher</option>
                          <option value="admin">Admin</option>
                        </select>
                      ) : (
                        <button
                          onClick={() => u.id !== user.id && setEditingRole(u.id)}
                          className={`px-2.5 py-1 rounded-full text-xs font-medium border capitalize ${ROLE_COLORS[u.role] || 'bg-gray-100 text-gray-600'} ${u.id !== user.id ? 'cursor-pointer hover:opacity-80' : 'cursor-default'}`}
                        >
                          {u.role}
                        </button>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${u.is_active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${u.is_active ? 'bg-green-500' : 'bg-red-500'}`} />
                        {u.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600">{u.document_count}</td>
                    <td className="px-4 py-3 text-gray-400 text-xs">
                      {u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3">
                      {u.id !== user.id && (
                        <div className="flex items-center justify-end gap-1">
                          <button
                            onClick={() => handleToggleActive(u.id)}
                            className={`p-1.5 rounded-lg transition ${u.is_active ? 'text-amber-500 hover:bg-amber-50' : 'text-green-500 hover:bg-green-50'}`}
                            title={u.is_active ? 'Deactivate' : 'Activate'}
                          >
                            {u.is_active ? <UserX className="w-4 h-4" /> : <UserCheck className="w-4 h-4" />}
                          </button>
                          <button
                            onClick={() => handleDeleteUser(u.id, u.username)}
                            className="p-1.5 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition"
                            title="Delete user"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Activity Tab */}
      {activeTab === 'activity' && stats && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Database className="w-4 h-4 text-gray-400" />
              System Overview
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { label: 'Active Users', value: stats.users.active, total: stats.users.total },
                { label: 'Processed Docs', value: stats.documents.processed, total: stats.documents.total },
                { label: 'Failed Docs', value: stats.documents.failed, total: stats.documents.total },
                { label: 'Chat Sessions', value: stats.chat.total_sessions },
                { label: 'Total Messages', value: stats.chat.total_messages },
                { label: 'Feedback Count', value: stats.feedback.total },
              ].map(({ label, value, total }) => (
                <div key={label} className="p-3 bg-gray-50 rounded-lg">
                  <p className="text-xs text-gray-500">{label}</p>
                  <p className="text-lg font-bold text-gray-900">
                    {value}{total != null && <span className="text-xs text-gray-400 font-normal"> / {total}</span>}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <Star className="w-4 h-4 text-amber-400" />
              Feedback Summary
            </h3>
            <p className="text-sm text-gray-600">
              {stats.feedback.total} total feedback submissions received from users.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
