/* ── QuizScanner Frontend JS ──────────────────────────────────────── */

let currentFile  = null;
let currentJobId = null;
let pollTimer    = null;

// ── Tab switching ──────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const id = tab.dataset.tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
    tab.classList.add('active');
    const panel = document.getElementById('tab-' + id);
    panel.classList.remove('hidden');
    panel.classList.add('active');
  });
});

// ── Single file upload & preview ──────────────────────────────────────
const fileInput  = document.getElementById('file-input');
const dropZone   = document.getElementById('drop-zone');
const previewBox = document.getElementById('preview-box');
const previewImg = document.getElementById('preview-img');
const actionRow  = document.getElementById('action-row');

fileInput.addEventListener('change', e => handleFileSelect(e.target.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());

function handleFileSelect(file) {
  if (!file) return;
  currentFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewBox.classList.remove('hidden');
    document.getElementById('preview-meta').textContent =
      `${file.name}  |  ${(file.size / 1024).toFixed(1)} KB`;
  };
  reader.readAsDataURL(file);

  actionRow.style.display = 'flex';
  clearResults();
}

// ── Core API calls ─────────────────────────────────────────────────────
async function runFullGrade() {
  if (!currentFile) return alert('Please upload an image first.');
  showProcessing('Running full pipeline (QR → OCR → Bubbles → Grade)…');
  try {
    const data = await postImage('/api/grade', currentFile);
    hideProcessing();
    if (data.success) {
      renderFullResults(data);
    } else {
      showError(data.error || 'Processing failed.');
    }
  } catch (err) {
    hideProcessing();
    showError(err.message);
  }
}

async function runQROnly() {
  if (!currentFile) return;
  showProcessing('Decoding QR code…');
  const data = await postImage('/api/decode-qr', currentFile);
  hideProcessing();
  renderQRResult(data);
}

async function runOCROnly() {
  if (!currentFile) return;
  showProcessing('Running OCR…');
  const data = await postImage('/api/extract-info', currentFile);
  hideProcessing();
  renderOCRResult(data);
}

async function runBubblesOnly() {
  if (!currentFile) return;
  showProcessing('Detecting bubbles…');
  const data = await postImage('/api/read-bubbles', currentFile);
  hideProcessing();
  renderBubbleResult(data);
}

async function postImage(url, file) {
  const fd = new FormData();
  fd.append('image', file);
  const resp = await fetch(url, { method: 'POST', body: fd });
  return resp.json();
}

// ── Result renderers ───────────────────────────────────────────────────
function renderFullResults(d) {
  const content = document.getElementById('results-content');
  document.getElementById('results-placeholder').classList.add('hidden');
  content.classList.remove('hidden');

  const r = d.report;
  const gradeClass = r ? 'grade-' + r.letter_grade[0] : 'grade-F';
  const negInfo = r && r.negative_marking > 0
    ? `<span class="incorrect">⚠ Negative marking: -${r.negative_marking}/wrong</span>` : '';

  let qrSection = '';
  if (d.qr_found) {
    const p1 = Object.entries(d.answer_key_p1 || {}).map(([q,a]) =>
      `<div class="answer-key-cell"><div class="q-num">${q}</div><div class="q-ans">${a}</div></div>`).join('');
    const p2 = Object.entries(d.answer_key_p2 || {}).map(([q,a]) =>
      `<div class="answer-key-cell"><div class="q-num">${q}</div><div class="q-ans">${a}</div></div>`).join('');
    qrSection = `
      <div class="qr-box">
        <strong>✅ QR Decoded</strong> — Set <strong>${d.quiz_set}</strong> | ${d.subject} ${negInfo}<br>
        <div style="margin-top:8px">
          <div style="font-size:11px;color:var(--muted);margin-bottom:4px">Part-I key</div>
          <div class="answer-key-grid">${p1}</div>
          <div style="font-size:11px;color:var(--muted);margin:8px 0 4px">Part-II key</div>
          <div class="answer-key-grid">${p2}</div>
        </div>
      </div>`;
  } else {
    qrSection = `<div class="result-card error">⚠ QR code not found — partial results only.</div>`;
  }

  let scoreSection = '';
  if (r) {
    scoreSection = `
      <div class="score-hero">
        <div class="grade-circle ${gradeClass}">
          <span class="grade-letter">${r.letter_grade}</span>
          <span class="grade-pct">${r.percentage.toFixed(1)}%</span>
        </div>
        <div class="score-details">
          <div class="score-big">${r.total_marks} / ${r.max_marks}</div>
          <div class="score-sub">Final Score for ${d.student_name || 'Student'}</div>
          <div class="score-stats">
            <span class="correct">✓ ${r.correct} correct</span>
            <span class="incorrect">✗ ${r.incorrect} incorrect</span>
            <span class="unattempt">— ${r.unattempted} blank</span>
            ${r.invalid > 0 ? `<span style="color:var(--amber-600)">⚠ ${r.invalid} invalid</span>` : ''}
          </div>
        </div>
      </div>`;
  }

  let infoSection = `
    <div class="info-row">
      <div class="info-item"><label>Student Name</label><div class="val">${d.student_name || '—'}</div></div>
      <div class="info-item"><label>Reg No</label><div class="val">${d.reg_no || '—'}</div></div>
    </div>`;

  let breakdownSection = '';
  if (r && r.breakdown) {
    const rows = r.breakdown.map(b => {
      const symClass = `sym-${b.status === 'unattempted' ? 'unattempt' : b.status}`;
      const rowBg = b.status === 'correct' ? 'style="background:rgba(22,163,74,.07)"'
                  : b.status === 'incorrect' ? 'style="background:rgba(220,38,38,.07)"' : '';
      return `<tr ${rowBg}>
        <td>${b.part}</td>
        <td><strong>${b.question}</strong></td>
        <td style="font-weight:700;color:var(--blue-600)">${b.student_answer || '—'}</td>
        <td style="font-weight:700;color:var(--green-600)">${b.correct_answer || '—'}</td>
        <td><span class="${symClass}">${b.symbol}</span></td>
        <td style="text-transform:capitalize">${b.status}</td>
        <td style="font-weight:700">${b.marks > 0 ? '+' : ''}${b.marks.toFixed(2)}</td>
      </tr>`;
    }).join('');

    breakdownSection = `
      <h3 style="font-size:14px;font-weight:600;margin:16px 0 8px">Per-Question Breakdown</h3>
      <table class="breakdown-table">
        <thead><tr><th>Part</th><th>Q</th><th>Your Ans</th><th>Key</th><th></th><th>Status</th><th>Marks</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <button class="btn-text" style="margin-top:10px" onclick='showFullReport(${JSON.stringify(d.text_report || "")})'>
        View full text report
      </button>`;
  }

  content.innerHTML = qrSection + infoSection + scoreSection + breakdownSection;
}

function renderQRResult(d) {
  const content = document.getElementById('results-content');
  document.getElementById('results-placeholder').classList.add('hidden');
  content.classList.remove('hidden');

  if (!d.success) { content.innerHTML = `<div class="result-card error">❌ ${d.error}</div>`; return; }

  const p1 = Object.entries(d.part1 || {}).map(([q,a]) =>
    `<div class="answer-key-cell"><div class="q-num">${q}</div><div class="q-ans">${a}</div></div>`).join('');
  const p2 = Object.entries(d.part2 || {}).map(([q,a]) =>
    `<div class="answer-key-cell"><div class="q-num">${q}</div><div class="q-ans">${a}</div></div>`).join('');

  content.innerHTML = `
    <div class="result-card success">
      <h3>✅ QR Code Decoded Successfully</h3>
      <div class="info-row">
        <div class="info-item"><label>Subject</label><div class="val">${d.subject}</div></div>
        <div class="info-item"><label>Set</label><div class="val">${d.quiz_set}</div></div>
        <div class="info-item"><label>Semester</label><div class="val">${d.semester || '—'}</div></div>
        <div class="info-item"><label>Negative Marking</label><div class="val">${d.negative_marking > 0 ? '-' + d.negative_marking + '/wrong' : 'None'}</div></div>
      </div>
      <div style="margin-top:8px">
        <div style="font-size:12px;font-weight:600;margin-bottom:4px">Part-I Answer Key</div>
        <div class="answer-key-grid">${p1}</div>
        <div style="font-size:12px;font-weight:600;margin:12px 0 4px">Part-II Answer Key</div>
        <div class="answer-key-grid">${p2}</div>
      </div>
      <div style="margin-top:14px;font-size:12px;color:var(--muted)">Raw: <code style="font-size:11px">${d.raw_payload}</code></div>
    </div>`;
}

function renderOCRResult(d) {
  const content = document.getElementById('results-content');
  document.getElementById('results-placeholder').classList.add('hidden');
  content.classList.remove('hidden');

  if (!d.success) { content.innerHTML = `<div class="result-card error">❌ ${d.error}</div>`; return; }

  const conf = (d.confidence * 100).toFixed(0);
  const confColor = d.confidence > 0.7 ? 'var(--green-600)' : d.confidence > 0.4 ? 'var(--amber-600)' : 'var(--red-600)';

  content.innerHTML = `
    <div class="result-card success">
      <h3>✅ Student Info Extracted</h3>
      <div class="info-row">
        <div class="info-item"><label>Student Name</label><div class="val">${d.name}</div></div>
        <div class="info-item"><label>Reg No</label><div class="val">${d.reg_no}</div></div>
        <div class="info-item"><label>Class</label><div class="val">${d.class_name || '—'}</div></div>
        <div class="info-item"><label>OCR Confidence</label><div class="val" style="color:${confColor}">${conf}%</div></div>
      </div>
    </div>`;
}

function renderBubbleResult(d) {
  const content = document.getElementById('results-content');
  document.getElementById('results-placeholder').classList.add('hidden');
  content.classList.remove('hidden');

  if (!d.success) { content.innerHTML = `<div class="result-card error">❌ ${d.error}</div>`; return; }

  const renderPart = (part, label) => {
    const cells = Object.entries(part).map(([q, a]) => {
      const key = `${label.replace('-','').toLowerCase()}_${q}`;
      const isInvalid   = d.invalid?.includes(key);
      const isUnattempt = d.unattempted?.includes(key);
      const bg = isInvalid ? '#FEF3C7' : isUnattempt ? 'var(--bg)' : 'var(--blue-50)';
      const tc = isInvalid ? '#92400E' : isUnattempt ? 'var(--muted)' : '#1E40AF';
      return `<div style="background:${bg};border:1px solid var(--border);border-radius:6px;padding:6px;text-align:center;">
        <div style="font-size:10px;color:var(--muted)">${q}</div>
        <div style="font-weight:700;font-size:16px;color:${tc}">${a || '—'}</div>
        ${isInvalid ? '<div style="font-size:9px;color:#92400E">INVALID</div>' : ''}
      </div>`;
    }).join('');
    return `<div style="margin-bottom:14px">
      <div style="font-size:13px;font-weight:600;margin-bottom:6px">${label}</div>
      <div style="display:grid;grid-template-columns:repeat(8,1fr);gap:4px">${cells}</div>
    </div>`;
  };

  content.innerHTML = `
    <div class="result-card success">
      <h3>✅ Bubble Sheet Read</h3>
      ${renderPart(d.part1, 'Part-I')}
      ${renderPart(d.part2, 'Part-II')}
      ${d.invalid?.length ? `<p style="color:var(--amber-600);font-size:13px">⚠ ${d.invalid.length} invalid bubble(s) detected</p>` : ''}
      ${d.unattempted?.length ? `<p style="color:var(--muted);font-size:13px">— ${d.unattempted.length} unattempted question(s)</p>` : ''}
    </div>`;
}

// ── Batch mode ─────────────────────────────────────────────────────────
const batchInput   = document.getElementById('batch-input');
const batchDropZone= document.getElementById('batch-drop-zone');
let batchFiles = [];

batchInput.addEventListener('change', e => handleBatchFiles(Array.from(e.target.files)));
batchDropZone.addEventListener('click', () => batchInput.click());
batchDropZone.addEventListener('dragover', e => { e.preventDefault(); batchDropZone.classList.add('drag-over'); });
batchDropZone.addEventListener('dragleave', () => batchDropZone.classList.remove('drag-over'));
batchDropZone.addEventListener('drop', e => {
  e.preventDefault();
  batchDropZone.classList.remove('drag-over');
  handleBatchFiles(Array.from(e.dataTransfer.files));
});

function handleBatchFiles(files) {
  batchFiles = files;
  const list = document.getElementById('batch-file-list');
  list.classList.remove('hidden');
  list.innerHTML = files.map(f =>
    `<div class="file-item">
      <span class="file-icon">📄</span>
      <span style="flex:1">${f.name}</span>
      <span style="color:var(--muted);font-size:12px">${(f.size/1024).toFixed(1)} KB</span>
    </div>`
  ).join('') + `<div style="font-size:13px;color:var(--muted);margin-top:8px">${files.length} file(s) ready</div>`;

  const btn = document.getElementById('btn-batch');
  btn.style.display = 'inline-flex';
  btn.textContent = `Process All ${files.length} Sheets & Generate Report`;
}

async function runBatch() {
  if (!batchFiles.length) return;

  const title = document.getElementById('batch-title').value || 'AI Quiz SP2026';
  const cls   = document.getElementById('batch-class').value  || 'BSE-4A';

  document.getElementById('batch-progress').classList.remove('hidden');
  document.getElementById('btn-batch').disabled = true;
  document.getElementById('batch-results').classList.add('hidden');

  const fd = new FormData();
  batchFiles.forEach(f => fd.append('images', f));
  fd.append('quiz_title', title);
  fd.append('class_name', cls);

  const resp = await fetch('/api/batch', { method: 'POST', body: fd });
  const data = await resp.json();

  if (!data.success) {
    showBatchError(data.error);
    return;
  }

  currentJobId = data.job_id;
  pollBatch();
}

function pollBatch() {
  pollTimer = setInterval(async () => {
    const resp = await fetch(`/api/batch/${currentJobId}`);
    const job  = await resp.json();

    const pct = job.total ? Math.round(job.processed / job.total * 100) : 0;
    document.getElementById('progress-bar').style.width = pct + '%';
    document.getElementById('progress-text').textContent =
      `${job.processed} / ${job.total} — ${job.current || ''}`;

    if (job.status === 'done') {
      clearInterval(pollTimer);
      document.getElementById('btn-batch').disabled = false;
      showBatchResults(job.result);
    } else if (job.status === 'error') {
      clearInterval(pollTimer);
      document.getElementById('btn-batch').disabled = false;
      showBatchError(job.error);
    }
  }, 1000);
}

function showBatchResults(result) {
  document.getElementById('progress-bar').style.width = '100%';
  document.getElementById('batch-results').classList.remove('hidden');
  const s = result.summary || {};
  document.getElementById('batch-summary').innerHTML = `
    <div class="info-row" style="margin-top:10px">
      <div class="info-item"><label>Students</label><div class="val">${s.total_students || '—'}</div></div>
      <div class="info-item"><label>Class Average</label><div class="val">${s.avg_percentage?.toFixed(1) || '—'}%</div></div>
      <div class="info-item"><label>Highest</label><div class="val">${s.max_percentage?.toFixed(1) || '—'}%</div></div>
      <div class="info-item"><label>Lowest</label><div class="val">${s.min_percentage?.toFixed(1) || '—'}%</div></div>
    </div>`;

  document.getElementById('dl-xlsx').onclick = () => downloadBatch('xlsx');
  document.getElementById('dl-csv').onclick  = () => downloadBatch('csv');
}

function showBatchError(err) {
  document.getElementById('batch-results').classList.remove('hidden');
  document.getElementById('batch-summary').innerHTML = `<div class="result-card error">❌ ${err}</div>`;
}

function downloadBatch(fmt) {
  if (currentJobId) {
    window.location.href = `/api/batch/${currentJobId}/download/${fmt}`;
  }
}

// ── Utilities ──────────────────────────────────────────────────────────
function showProcessing(msg) {
  document.getElementById('processing').classList.remove('hidden');
  document.getElementById('processing-text').textContent = msg;
  document.getElementById('results-placeholder').classList.add('hidden');
  document.getElementById('results-content').classList.add('hidden');
}

function hideProcessing() {
  document.getElementById('processing').classList.add('hidden');
}

function showError(msg) {
  const content = document.getElementById('results-content');
  document.getElementById('results-placeholder').classList.add('hidden');
  content.classList.remove('hidden');
  content.innerHTML = `<div class="result-card error"><h3>❌ Error</h3><p>${msg}</p></div>`;
}

function clearResults() {
  document.getElementById('results-placeholder').classList.remove('hidden');
  document.getElementById('results-content').classList.add('hidden');
  document.getElementById('results-content').innerHTML = '';
}

function showFullReport(text) {
  document.getElementById('modal-text').textContent = text;
  document.getElementById('modal-overlay').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});
