import { useCallback, useEffect, useRef, useState } from "react";
import { deleteJSON, fetchJSON, postForm } from "../lib/api.js";

/**
 * Documents.jsx
 * -------------
 * Week 2 — the document library: upload PDFs into the knowledge base and
 * remove them again.
 *
 * Consumes:
 *   POST   /documents/upload/                 -> extract, chunk, embed, store
 *   GET    /documents/files                   -> one row per ingested file
 *   DELETE /documents/by-file/:file_name
 *
 * Upload runs the whole ingestion pipeline synchronously, including the
 * embedding calls, so a large PDF on a rate-limited API key can take a minute
 * or more; the page says so instead of looking stuck.
 */

const MAX_MB = 10; // Mirrors MAX_FILE_SIZE_MB in backend/app/main.py.

const panel = "rounded-md border border-[#2A3B36] bg-[#16211E]";

function UploadZone({ onUploaded }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const choose = (candidate) => {
    setResult(null);
    setError(null);
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    if (candidate.size > MAX_MB * 1024 * 1024) {
      setError(`That file is ${(candidate.size / 1024 / 1024).toFixed(1)}MB; the limit is ${MAX_MB}MB.`);
      return;
    }
    setFile(candidate);
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await postForm("/documents/upload/", form);
      const pages = new Set(data.chunks.map((c) => c.page_number)).size;
      setResult(`${file.name}: ${data.chunks.length} chunks from ${pages} page${pages === 1 ? "" : "s"} stored.`);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      onUploaded();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className={`${panel} p-5`}>
      <div
        role="button"
        tabIndex={0}
        onClick={() => !uploading && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !uploading && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!uploading) choose(e.dataTransfer.files?.[0]);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center rounded border border-dashed px-6 py-10 text-center transition-colors ${
          dragging ? "border-[#5FE0B8] bg-[#152820]" : "border-[#3A4C46] hover:border-[#5F726B]"
        }`}
      >
        <p className="text-[14px] text-[#E8EDE9]">
          {file ? file.name : "Drop a PDF here, or click to choose one"}
        </p>
        <p className="mt-1 text-[12px] text-[#5F726B]">
          {file ? `${(file.size / 1024 / 1024).toFixed(2)}MB` : `Text-based PDFs up to ${MAX_MB}MB. Scanned PDFs need OCR first.`}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => choose(e.target.files?.[0])}
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={upload}
          disabled={!file || uploading}
          className="rounded bg-[#5FE0B8] px-5 py-2 text-[13px] font-semibold text-[#0F1614] hover:bg-[#7CE0B8] disabled:opacity-40"
        >
          {uploading ? "Processing…" : "Upload and index"}
        </button>
        {uploading && (
          <span className="text-[12px] text-[#8FA39B]">
            Extracting, chunking and embedding. Large PDFs can take a minute or more on a rate-limited API key.
          </span>
        )}
      </div>

      {result && (
        <div className="mt-4 rounded border border-[#2F5C48] bg-[#152820] px-4 py-3 text-[12px] text-[#7CE0B8]">{result}</div>
      )}
      {error && (
        <div className="mt-4 rounded border border-[#5C2F2F] bg-[#281616] px-4 py-3 text-[12px] text-[#E37B6B]">{error}</div>
      )}
    </div>
  );
}

function FileList({ files, loading, error, onDelete }) {
  const [confirming, setConfirming] = useState(null);
  const [deleting, setDeleting] = useState(null);

  const remove = async (name) => {
    setDeleting(name);
    try {
      await onDelete(name);
    } finally {
      setDeleting(null);
      setConfirming(null);
    }
  };

  return (
    <div className={`${panel} overflow-hidden`}>
      <div className="border-b border-[#2A3B36] px-4 py-3">
        <h2 className="text-[14px] text-[#E8EDE9]">Knowledge base</h2>
      </div>
      {loading ? (
        <p className="p-4 text-[12px] text-[#8FA39B]">Loading…</p>
      ) : error ? (
        <p className="p-4 text-[12px] text-[#E37B6B]">{error}</p>
      ) : files.length === 0 ? (
        <p className="p-4 text-[12px] text-[#8FA39B]">
          Nothing indexed yet. Upload a PDF to start asking questions about it.
        </p>
      ) : (
        <ul>
          {files.map((f) => (
            <li
              key={`${f.is_file}-${f.file_name}`}
              className="flex flex-wrap items-center justify-between gap-3 border-b border-[#20302B] px-4 py-3 last:border-b-0"
            >
              <div className="min-w-0">
                <p className="truncate text-[13px] text-[#E8EDE9]" title={f.file_name}>
                  {f.file_name}
                </p>
                <p className="font-mono text-[11px] text-[#5F726B]">
                  {f.chunks} chunks{f.is_file ? ` · ${f.pages} page${f.pages === 1 ? "" : "s"}` : " · pasted text"}
                </p>
              </div>
              {f.is_file &&
                (confirming === f.file_name ? (
                  <span className="flex items-center gap-2 text-[12px]">
                    <span className="text-[#8FA39B]">Delete all {f.chunks} chunks?</span>
                    <button
                      type="button"
                      onClick={() => remove(f.file_name)}
                      disabled={deleting === f.file_name}
                      className="rounded border border-[#5C2F2F] px-2 py-1 text-[#E37B6B] hover:bg-[#281616] disabled:opacity-50"
                    >
                      {deleting === f.file_name ? "Deleting…" : "Delete"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirming(null)}
                      className="px-2 py-1 text-[#8FA39B] hover:text-[#E8EDE9]"
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    onClick={() => setConfirming(f.file_name)}
                    className="text-[12px] text-[#8FA39B] hover:text-[#E37B6B]"
                  >
                    Remove
                  </button>
                ))}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function Documents() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setFiles(await fetchJSON("/documents/files"));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const remove = async (name) => {
    try {
      await deleteJSON(`/documents/by-file/${encodeURIComponent(name)}`);
    } catch (err) {
      setError(err.message);
    }
    await load();
  };

  return (
    <div className="min-h-screen bg-[#0F1614] px-6 py-8 md:px-10">
      <div className="mx-auto max-w-4xl">
        <header className="mb-8">
          <p className="mb-1 text-[13px] text-[#5F726B]">RAGOps · Documents</p>
          <h1 className="text-2xl text-[#E8EDE9]">Build the knowledge base</h1>
          <p className="mt-1 text-[13px] text-[#8FA39B]">
            Each PDF is extracted, cleaned, split into overlapping chunks and embedded into pgvector, with its file
            name and page numbers kept for citations.
          </p>
        </header>
        <div className="space-y-6">
          <UploadZone onUploaded={load} />
          <FileList files={files} loading={loading} error={error} onDelete={remove} />
        </div>
      </div>
    </div>
  );
}
