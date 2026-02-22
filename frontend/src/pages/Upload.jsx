/**
 * Upload Page - PDF document upload with drag-and-drop support.
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { documentsAPI } from '../services/api';
import { Upload as UploadIcon, File, CheckCircle, XCircle, Loader } from 'lucide-react';
import toast from 'react-hot-toast';

export default function Upload() {
  const [uploads, setUploads] = useState([]);

  const onDrop = useCallback(async (acceptedFiles) => {
    for (const file of acceptedFiles) {
      // Validate client-side
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        toast.error(`${file.name} is not a PDF file`);
        continue;
      }

      const uploadEntry = {
        id: Date.now() + Math.random(),
        name: file.name,
        size: file.size,
        status: 'uploading',
        progress: 0,
      };

      setUploads((prev) => [uploadEntry, ...prev]);

      try {
        await documentsAPI.upload(file, (event) => {
          const progress = Math.round((event.loaded * 100) / event.total);
          setUploads((prev) =>
            prev.map((u) =>
              u.id === uploadEntry.id ? { ...u, progress } : u
            )
          );
        });

        setUploads((prev) =>
          prev.map((u) =>
            u.id === uploadEntry.id
              ? { ...u, status: 'success', progress: 100 }
              : u
          )
        );
        toast.success(`${file.name} uploaded successfully!`);
      } catch (err) {
        const msg = err.response?.data?.detail || 'Upload failed';
        setUploads((prev) =>
          prev.map((u) =>
            u.id === uploadEntry.id
              ? { ...u, status: 'error', error: msg }
              : u
          )
        );
        toast.error(`Failed to upload ${file.name}`);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true,
  });

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Documents</h1>
        <p className="text-gray-500 mt-1">Upload PDF files for intelligent analysis and Q&A</p>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <UploadIcon className={`w-12 h-12 mx-auto mb-4 ${isDragActive ? 'text-primary-500' : 'text-gray-400'}`} />
        <p className="text-lg font-medium text-gray-700">
          {isDragActive ? 'Drop your PDFs here...' : 'Drag & drop PDF files here'}
        </p>
        <p className="text-gray-500 mt-1">or click to browse files</p>
        <p className="text-gray-400 text-sm mt-3">PDF files only, up to 50MB each</p>
      </div>

      {/* Upload List */}
      {uploads.length > 0 && (
        <div className="mt-8 space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Uploads</h2>
          {uploads.map((upload) => (
            <div
              key={upload.id}
              className="flex items-center gap-4 p-4 bg-white rounded-xl border border-gray-200 shadow-sm"
            >
              <File className="w-8 h-8 text-red-500 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{upload.name}</p>
                <p className="text-xs text-gray-500">{formatSize(upload.size)}</p>
                {upload.status === 'uploading' && (
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="bg-primary-600 h-1.5 rounded-full transition-all"
                      style={{ width: `${upload.progress}%` }}
                    />
                  </div>
                )}
                {upload.error && (
                  <p className="text-xs text-red-500 mt-1">{upload.error}</p>
                )}
              </div>
              <div className="shrink-0">
                {upload.status === 'uploading' && (
                  <Loader className="w-5 h-5 text-primary-500 animate-spin" />
                )}
                {upload.status === 'success' && (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                )}
                {upload.status === 'error' && (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
