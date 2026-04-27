/**
 * Trauma Lifesaver — Medical Dashboard UI
 */

document.addEventListener('DOMContentLoaded', () => {

    // ── Theme toggle ───────────────────────────────────────────────────
    const themeToggle = document.getElementById('themeToggle');
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
    });

    let currentSessionId = null;
    let qaStreaming      = false;
    let stagedFiles      = null;
    let previewUrls      = [];

    const el = {
        fileInput:          document.getElementById('fileInput'),
        dropTarget:         document.getElementById('dropTarget'),
        fileCountBadge:     document.getElementById('fileCountBadge'),
        fileCountText:      document.getElementById('fileCountText'),
        ctThumbnails:       document.getElementById('ctThumbnails'),
        ctResultsStrip:     document.getElementById('ctResultsStrip'),
        fileValidation:     document.getElementById('fileValidation'),
        analyzeBtn:         document.getElementById('analyzeBtn'),
        analyzeHint:        document.getElementById('analyzeHint'),
        metaPatientId:      document.getElementById('metaPatientId'),
        metaAge:            document.getElementById('metaAge'),
        metaState:          document.getElementById('metaState'),
        clinicalNotes:      document.getElementById('clinicalNotes'),
        vitHR:              document.getElementById('vitHR'),
        vitBP:              document.getElementById('vitBP'),
        vitGCS:             document.getElementById('vitGCS'),
        loadingOverlay:     document.getElementById('loadingOverlay'),
        errorBanner:        document.getElementById('errorBanner'),
        errorMessage:       document.getElementById('errorMessage'),
        errorDismiss:       document.getElementById('errorDismiss'),
        resultsContainer:   document.getElementById('resultsContainer'),
        patientIdDisplay:   document.getElementById('patientIdDisplay'),
        patientMetaChips:   document.getElementById('patientMetaChips'),
        riskBanner:         document.getElementById('riskBanner'),
        riskValue:          document.getElementById('riskValue'),
        riskVolume:         document.getElementById('riskVolume'),
        riskEAST:           document.getElementById('riskEAST'),
        triageRows:         document.getElementById('triageRows'),
        volume:             document.getElementById('volume'),
        pixels:             document.getElementById('pixels'),
        severity:           document.getElementById('severity'),
        organs:             document.getElementById('organs'),
        injuryPattern:      document.getElementById('injuryPattern'),
        diffList:           document.getElementById('diffList'),
        llmReport:          document.getElementById('llmReport'),
        copyReportBtn:      document.getElementById('copyReportBtn'),
        chatHistory:        document.getElementById('chatHistory'),
        typingIndicator:    document.getElementById('typingIndicator'),
        qaInput:            document.getElementById('qaInput'),
        qaSubmit:           document.getElementById('qaSubmit'),
        resetBtn:           document.getElementById('resetBtn'),
    };

    // ── File staging ───────────────────────────────────────────────────
    el.fileInput.addEventListener('change', (e) => stageFiles(e.target.files));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
        el.dropTarget.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
    );

    el.dropTarget.addEventListener('dragover',  () => el.dropTarget.classList.add('dragover'));
    el.dropTarget.addEventListener('dragleave', () => el.dropTarget.classList.remove('dragover'));
    el.dropTarget.addEventListener('drop', (e) => {
        el.dropTarget.classList.remove('dragover');
        stageFiles(e.dataTransfer.files);
    });

    function stageFiles(files) {
        if (!files || files.length === 0) return;

        const valid   = Array.from(files).filter(f => f.type.startsWith('image/'));
        const invalid = files.length - valid.length;

        if (invalid > 0) showValidation(`${invalid} unsupported file(s) skipped — PNG/JPG only`);
        else hideValidation();

        if (valid.length === 0) {
            el.analyzeBtn.disabled = true;
            el.analyzeHint.textContent = 'No valid image files selected';
            return;
        }

        stagedFiles  = valid;
        previewUrls  = new Array(valid.length).fill(null);

        const n = valid.length;
        el.fileCountText.textContent = `${n} file${n !== 1 ? 's' : ''} staged`;
        el.fileCountBadge.classList.remove('hidden');
        el.analyzeBtn.disabled = false;
        el.analyzeHint.textContent = `${n} CT slice${n !== 1 ? 's' : ''} ready`;

        // Generate thumbnails with delete buttons
        el.ctThumbnails.innerHTML = '';
        valid.forEach((file, i) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewUrls[i] = e.target.result;
                const wrap = document.createElement('div');
                wrap.className = 'ct-thumb-wrap';

                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'ct-thumb';
                img.title = file.name;

                const del = document.createElement('button');
                del.className = 'ct-thumb-delete';
                del.innerHTML = '<i class="fas fa-times"></i>';
                del.title = 'Remove slice';
                del.addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    stagedFiles = stagedFiles.filter((_, j) => j !== i);
                    previewUrls[i] = null;
                    wrap.remove();
                    const remaining = el.ctThumbnails.querySelectorAll('.ct-thumb-wrap').length;
                    if (remaining === 0) {
                        el.ctThumbnails.classList.add('hidden');
                        el.fileCountBadge.classList.add('hidden');
                        el.analyzeBtn.disabled = true;
                        el.analyzeHint.textContent = 'Select CT slices to begin';
                    } else {
                        el.fileCountText.textContent = `${remaining} file${remaining !== 1 ? 's' : ''} staged`;
                        el.analyzeHint.textContent = `${remaining} CT slice${remaining !== 1 ? 's' : ''} ready`;
                    }
                });

                wrap.appendChild(img);
                wrap.appendChild(del);
                el.ctThumbnails.appendChild(wrap);
            };
            reader.readAsDataURL(file);
        });
        el.ctThumbnails.classList.remove('hidden');
    }

    // ── Error / validation banners ─────────────────────────────────────
    function showError(msg) {
        el.errorMessage.textContent = msg;
        el.errorBanner.classList.remove('hidden');
        el.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() { el.errorBanner.classList.add('hidden'); }

    el.errorDismiss.addEventListener('click', hideError);

    function showValidation(msg) {
        el.fileValidation.textContent = msg;
        el.fileValidation.classList.remove('hidden');
    }

    function hideValidation() { el.fileValidation.classList.add('hidden'); }

    // ── Loading steps ──────────────────────────────────────────────────
    const STEP_IDS = ['step1','step2','step3','step4','step5'];
    let stepTimer = null, currentStep = 0;

    function setStepState(id, state) {
        const s = document.getElementById(id);
        if (!s) return;
        s.className = `step ${state}`;
        const icon = s.querySelector('.step-icon');
        if (!icon) return;
        if (state === 'done')   { icon.className = 'step-icon done';    icon.innerHTML = '<i class="fas fa-check"></i>'; }
        if (state === 'active') { icon.className = 'step-icon active';  icon.innerHTML = '<i class="fas fa-circle-notch"></i>'; }
        if (state === 'pending'){ icon.className = 'step-icon pending'; icon.innerHTML = '<i class="fas fa-circle"></i>'; }
    }

    function startLoadingSteps() {
        currentStep = 0;
        STEP_IDS.forEach(id => setStepState(id, 'pending'));
        setStepState(STEP_IDS[0], 'active');
        stepTimer = setInterval(() => {
            if (currentStep < STEP_IDS.length - 1) {
                setStepState(STEP_IDS[currentStep], 'done');
                currentStep++;
                setStepState(STEP_IDS[currentStep], 'active');
            }
        }, 3500);
    }

    function stopLoadingSteps() {
        clearInterval(stepTimer);
        STEP_IDS.forEach(id => setStepState(id, 'done'));
    }

    // ── Analyze ────────────────────────────────────────────────────────
    el.analyzeBtn.addEventListener('click', async () => {
        if (!stagedFiles || stagedFiles.length === 0) return;
        hideError();
        await runAnalysis(stagedFiles);
    });

    async function runAnalysis(files) {
        el.loadingOverlay.classList.add('active');
        el.resultsContainer.classList.add('hidden');
        startLoadingSteps();

        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        if (el.metaPatientId.value.trim()) formData.append('patient_id',     el.metaPatientId.value.trim());
        if (el.metaAge.value.trim())       formData.append('age',            el.metaAge.value.trim());
        if (el.metaState.value)            formData.append('clinical_state', el.metaState.value);
        if (el.clinicalNotes && el.clinicalNotes.value.trim()) formData.append('clinical_notes', el.clinicalNotes.value.trim());
        if (el.vitHR.value.trim())  formData.append('hr',  el.vitHR.value.trim());
        if (el.vitBP.value.trim())  formData.append('bp',  el.vitBP.value.trim());
        if (el.vitGCS.value.trim()) formData.append('gcs', el.vitGCS.value.trim());

        try {
            // Step 1: submit job — returns immediately with job_id
            const resp = await fetch('/upload', { method: 'POST', body: formData });
            const init = await resp.json();
            if (!init.job_id) throw new Error(init.error || 'Failed to start analysis');

            // Step 2: poll /status/<job_id> every 2s
            await pollJob(init.job_id);
        } catch (err) {
            stopLoadingSteps();
            el.loadingOverlay.classList.remove('active');
            showError(`Analysis failed: ${err.message}`);
        }
    }

    async function pollJob(jobId) {
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const resp = await fetch(`/status/${jobId}`);
                    const data = await resp.json();
                    if (data.status === 'done') {
                        stopLoadingSteps();
                        el.loadingOverlay.classList.remove('active');
                        renderResults(data.result);
                        resolve();
                    } else if (data.status === 'error') {
                        stopLoadingSteps();
                        el.loadingOverlay.classList.remove('active');
                        reject(new Error(data.error || 'Pipeline error'));
                    } else {
                        setTimeout(poll, 2000); // still processing — poll again
                    }
                } catch (err) {
                    stopLoadingSteps();
                    el.loadingOverlay.classList.remove('active');
                    reject(err);
                }
            };
            poll();
        });
    }

    // ── Render results ─────────────────────────────────────────────────
    function renderResults(result) {
        currentSessionId = result.session_id;
        el.resultsContainer.classList.remove('hidden');
        el.resultsContainer.scrollIntoView({ behavior: 'smooth' });

        el.patientIdDisplay.textContent = result.patient_id || 'CASE-UNKNOWN';

        // Risk banner
        const quant       = result.quantification || {};
        const riskLevel   = (quant.risk_level || 'UNKNOWN').toUpperCase();
        const volumeML    = (quant.volume_ml || 0).toFixed(1);
        const rec         = quant.recommendation || '--';

        el.riskBanner.className    = `risk-banner ${riskLevel.toLowerCase()}`;
        el.riskValue.textContent   = riskLevel;
        el.riskVolume.textContent  = `${volumeML} mL`;
        el.riskEAST.textContent    = rec;

        // CT images in results sidebar
        el.ctResultsStrip.innerHTML = '';
        const scores = result.triage?.per_slice_scores || [];
        previewUrls.forEach((src, i) => {
            if (!src) return;
            const suspicious = (scores[i] || 0) >= 0.25;
            const img = document.createElement('img');
            img.src = src;
            img.className = `ct-result-thumb${suspicious ? ' suspicious' : ''}`;
            img.title = `Slice ${i + 1} — ${suspicious ? 'Suspicious' : 'Clear'}`;
            el.ctResultsStrip.appendChild(img);
        });

        // Triage rows
        el.triageRows.innerHTML = '';
        scores.forEach((score, i) => {
            const pct        = Math.round(score * 100);
            const suspicious = score >= 0.25;
            const cls        = suspicious ? 'suspicious' : 'clear';
            const label      = suspicious ? 'Suspicious' : 'Clear';

            const row = document.createElement('div');
            row.className = 'triage-row';
            row.innerHTML = `
                <span class="triage-slice-label">SL-${String(i + 1).padStart(2, '0')}</span>
                <div class="triage-bar-wrap">
                    <div class="triage-bar-bg">
                        <div class="triage-bar-fill ${cls}" style="width:${pct}%"></div>
                    </div>
                    <span class="triage-pct">${pct}%</span>
                </div>
                <span class="triage-status ${cls}">${label}</span>
            `;
            el.triageRows.appendChild(row);
        });

        // Quantification
        el.volume.textContent = `${volumeML} mL`;
        el.pixels.textContent = (quant.num_voxels || 0).toLocaleString();

        // Visual findings
        const vf = result.visual_findings || {};
        el.severity.textContent      = (vf.severity_estimate || '--').toUpperCase();
        const organs = vf.organs_involved || [];
        el.organs.textContent        = organs.length ? organs.join(', ') : 'None identified';
        el.injuryPattern.textContent = vf.injury_pattern || '--';

        // Differential
        el.diffList.innerHTML = '';
        const diffs = vf.differential_diagnosis || [];
        if (diffs.length) {
            diffs.forEach(d => {
                const li = document.createElement('li');
                li.textContent = d;
                el.diffList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No differential available.';
            el.diffList.appendChild(li);
        }

        // Report — render as lightweight HTML (bold headings, no full markdown lib needed)
        const rawReport = result.report || '--';
        el.llmReport.innerHTML = rawReport
            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/^(CLINICAL INDICATION|FINDINGS|AAST GRADING|IMPRESSION|EAST RECOMMENDATION|LABS & IMAGING|LABS & FOLLOW-UP)(.*?)$/gm,
                '<h2>$1$2</h2>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');

        // Patient meta chips
        el.patientMetaChips.innerHTML = '';
        const age   = el.metaAge.value.trim();
        const state = el.metaState.value;
        const admTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        [age ? `Age: ${age} yrs` : null, state || null, `Admitted: ${admTime}`]
            .filter(Boolean)
            .forEach(txt => {
                const span = document.createElement('span');
                span.className = 'meta-chip';
                span.textContent = txt;
                el.patientMetaChips.appendChild(span);
            });

        // Reset chat
        el.chatHistory.innerHTML = `
            <div class="message ai">
                <div class="msg-avatar"><i class="fas fa-robot"></i></div>
                <div class="msg-body">
                    <div class="msg-role">Trauma AI</div>
                    <div class="msg-text">Scan analyzed. Ask me anything about the findings or treatment options.</div>
                </div>
            </div>`;
    }

    // ── Copy report ────────────────────────────────────────────────────
    el.copyReportBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(el.llmReport.textContent);
            el.copyReportBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
            el.copyReportBtn.classList.add('copied');
            setTimeout(() => {
                el.copyReportBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                el.copyReportBtn.classList.remove('copied');
            }, 2000);
        } catch {
            showError('Clipboard write blocked — please select and copy manually.');
        }
    });

    // ── Markdown renderer ──────────────────────────────────────────────
    function renderMarkdown(text) {
        return text
            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/^#{3}\s+(.+)$/gm, '<strong>$1</strong>')
            .replace(/^#{1,2}\s+(.+)$/gm, '<strong style="font-size:1em">$1</strong>')
            .replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>')
            .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
    }

    // ── Q&A ────────────────────────────────────────────────────────────

    el.qaSubmit.addEventListener('click', submitQuestion);
    el.qaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuestion(); }
    });

    function submitQuestion() {
        if (qaStreaming) return;
        const q = el.qaInput.value.trim();
        if (!q) return;
        if (!currentSessionId) { showError('Please analyze a scan first.'); return; }

        appendMessage('You', q, 'user');
        el.qaInput.value = '';
        const aiTextEl = appendMessage('Trauma AI', '', 'ai');
        let aiRawText = '';
        el.qaSubmit.disabled = true;
        el.typingIndicator.classList.remove('hidden');
        qaStreaming = true;

        const url = `/qa-stream?session_id=${encodeURIComponent(currentSessionId)}&q=${encodeURIComponent(q)}`;
        const evtSource = new EventSource(url);

        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                evtSource.close();
                el.qaSubmit.disabled = false;
                el.typingIndicator.classList.add('hidden');
                qaStreaming = false;
                // Render full markdown now that stream is complete
                aiTextEl.innerHTML = renderMarkdown(aiRawText);
                scrollChat();
                return;
            }
            el.typingIndicator.classList.add('hidden');
            // Show raw text while streaming (markdown on partial tokens breaks formatting)
            aiRawText += e.data;
            aiTextEl.textContent = aiRawText;
            scrollChat();
        };

        evtSource.onerror = () => {
            evtSource.close();
            el.qaSubmit.disabled = false;
            el.typingIndicator.classList.add('hidden');
            qaStreaming = false;
            aiTextEl.textContent += ' [Connection Error]';
        };
    }

    function appendMessage(role, text, type) {
        const isAI = type === 'ai';
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = isAI ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

        const body = document.createElement('div');
        body.className = 'msg-body';

        const roleDiv = document.createElement('div');
        roleDiv.className = 'msg-role';
        roleDiv.textContent = role;

        const textDiv = document.createElement('div');
        textDiv.className = 'msg-text';
        textDiv.textContent = text;

        body.appendChild(roleDiv);
        body.appendChild(textDiv);
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(body);
        el.chatHistory.appendChild(msgDiv);
        scrollChat();
        return textDiv;
    }

    function scrollChat() {
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
    }

    // ── Reset ──────────────────────────────────────────────────────────
    el.resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        el.fileInput.value   = '';
        stagedFiles          = null;
        previewUrls          = [];
        currentSessionId     = null;

        el.fileCountBadge.classList.add('hidden');
        el.ctThumbnails.innerHTML = '';
        el.ctThumbnails.classList.add('hidden');
        el.analyzeBtn.disabled        = true;
        el.analyzeHint.textContent    = 'Select CT slices to begin';

        el.vitHR.value  = '';
        el.vitBP.value  = '';
        el.vitGCS.value = '';
        if (el.metaPatientId) el.metaPatientId.value = '';
        if (el.metaAge)       el.metaAge.value = '';
        if (el.metaState)     el.metaState.value = '';
        if (el.clinicalNotes) el.clinicalNotes.value = '';

        el.resultsContainer.classList.add('hidden');
        hideError();
        hideValidation();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
