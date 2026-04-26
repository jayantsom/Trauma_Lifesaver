document.addEventListener('DOMContentLoaded', () => {

    const themeToggle = document.getElementById('themeToggle');
    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.contains('dark-mode');
        document.body.classList.toggle('dark-mode', !isDark);
        document.body.classList.toggle('light-mode', isDark);
    });

    let currentSessionId = null;
    let qaStreaming = false;
    let stagedFiles = null;
    let previewUrls = [];

    const el = {
        fileInput: document.getElementById('fileInput'),
        dropTarget: document.getElementById('dropTarget'),
        fileCountBadge: document.getElementById('fileCountBadge'),
        fileCountText: document.getElementById('fileCountText'),
        ctThumbnails: document.getElementById('ctThumbnails'),
        ctResultsStrip: document.getElementById('ctResultsStrip'),
        fileValidation: document.getElementById('fileValidation'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        analyzeHint: document.getElementById('analyzeHint'),
        vitHR: document.getElementById('vitHR'),
        vitBP: document.getElementById('vitBP'),
        vitGCS: document.getElementById('vitGCS'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        errorBanner: document.getElementById('errorBanner'),
        errorMessage: document.getElementById('errorMessage'),
        errorDismiss: document.getElementById('errorDismiss'),
        resultsContainer: document.getElementById('resultsContainer'),
        patientIdDisplay: document.getElementById('patientIdDisplay'),
        riskBanner: document.getElementById('riskBanner'),
        riskValue: document.getElementById('riskValue'),
        riskVolume: document.getElementById('riskVolume'),
        riskEAST: document.getElementById('riskEAST'),
        triageRows: document.getElementById('triageRows'),
        volume: document.getElementById('volume'),
        pixels: document.getElementById('pixels'),
        severity: document.getElementById('severity'),
        organs: document.getElementById('organs'),
        injuryPattern: document.getElementById('injuryPattern'),
        diffList: document.getElementById('diffList'),
        llmReport: document.getElementById('llmReport'),
        copyReportBtn: document.getElementById('copyReportBtn'),
        chatHistory: document.getElementById('chatHistory'),
        typingIndicator: document.getElementById('typingIndicator'),
        qaInput: document.getElementById('qaInput'),
        qaSubmit: document.getElementById('qaSubmit'),
        resetBtn: document.getElementById('resetBtn'),
    };

    el.fileInput.addEventListener('change', (e) => stageFiles(e.target.files));
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
        el.dropTarget.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
    );
    el.dropTarget.addEventListener('drop', (e) => {
        el.dropTarget.classList.remove('dragover');
        stageFiles(e.dataTransfer.files);
    });

    function stageFiles(files) {
        if (!files || files.length === 0) return;
        const valid = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (valid.length === 0) return;

        stagedFiles = valid;
        previewUrls = new Array(valid.length).fill(null);
        el.fileCountText.textContent = `${valid.length} files staged`;
        el.fileCountBadge.classList.remove('hidden');
        el.analyzeBtn.disabled = false;
        el.analyzeHint.textContent = `${valid.length} CT slices ready`;

        el.ctThumbnails.innerHTML = '';
        valid.forEach((file, i) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewUrls[i] = e.target.result;
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'ct-thumb';
                el.ctThumbnails.appendChild(img);
            };
            reader.readAsDataURL(file);
        });
        el.ctThumbnails.classList.remove('hidden');
    }

    const STEP_IDS = ['step1','step2','step3','step4','step5'];
    let stepTimer = null, currentStep = 0;

    function startLoadingSteps() {
        currentStep = 0;
        STEP_IDS.forEach(id => document.getElementById(id).className = 'step');
        document.getElementById(STEP_IDS[0]).classList.add('active');
        stepTimer = setInterval(() => {
            if (currentStep < STEP_IDS.length - 1) {
                document.getElementById(STEP_IDS[currentStep]).className = 'step done';
                currentStep++;
                document.getElementById(STEP_IDS[currentStep]).classList.add('active');
            }
        }, 3500);
    }

    function stopLoadingSteps() {
        clearInterval(stepTimer);
        STEP_IDS.forEach(id => document.getElementById(id).className = 'step done');
    }

    el.analyzeBtn.addEventListener('click', async () => {
        if (!stagedFiles || stagedFiles.length === 0) return;
        
        el.loadingOverlay.classList.add('active');
        el.resultsContainer.classList.add('hidden');
        startLoadingSteps();

        const formData = new FormData();
        stagedFiles.forEach(f => formData.append('files', f));
        if (el.vitHR.value.trim()) formData.append('hr', el.vitHR.value.trim());
        if (el.vitBP.value.trim()) formData.append('bp', el.vitBP.value.trim());
        if (el.vitGCS.value.trim()) formData.append('gcs', el.vitGCS.value.trim());

        try {
            const resp = await fetch('/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            stopLoadingSteps();
            el.loadingOverlay.classList.remove('active');
            if (data.success) renderResults(data.result);
            else alert(data.error);
        } catch (err) {
            stopLoadingSteps();
            el.loadingOverlay.classList.remove('active');
            alert(err.message);
        }
    });

    function renderResults(result) {
        currentSessionId = result.session_id;
        el.resultsContainer.classList.remove('hidden');
        el.patientIdDisplay.textContent = result.patient_id || 'UNKNOWN';

        const quant = result.quantification || {};
        const riskLevel = (quant.risk_level || 'UNKNOWN').toUpperCase();
        el.riskBanner.className = `risk-banner ${riskLevel.toLowerCase()}`;
        el.riskValue.textContent = riskLevel;
        el.riskVolume.textContent = `${(quant.volume_ml || 0).toFixed(1)} mL`;
        el.riskEAST.textContent = quant.recommendation || '--';

        el.ctResultsStrip.innerHTML = '';
        const scores = result.triage?.per_slice_scores || [];
        previewUrls.forEach((src, i) => {
            if (!src) return;
            const img = document.createElement('img');
            img.src = src;
            img.className = `ct-result-thumb ${(scores[i] || 0) >= 0.25 ? 'suspicious' : ''}`;
            el.ctResultsStrip.appendChild(img);
        });

        el.triageRows.innerHTML = '';
        scores.forEach((score, i) => {
            const pct = Math.round(score * 100);
            const suspicious = score >= 0.25;
            const cls = suspicious ? 'suspicious' : 'clear';
            el.triageRows.innerHTML += `
                <div class="triage-row">
                    <span>SL-${String(i+1).padStart(2,'0')}</span>
                    <div class="triage-bar-wrap">
                        <div class="triage-bar-bg"><div class="triage-bar-fill ${cls}" style="width:${pct}%"></div></div>
                        <span>${pct}%</span>
                    </div>
                    <span class="triage-status ${cls}">${suspicious ? 'Susp' : 'Clear'}</span>
                </div>
            `;
        });

        el.volume.textContent = `${(quant.volume_ml || 0).toFixed(1)} mL`;
        el.pixels.textContent = quant.num_voxels || 0;

        const vf = result.visual_findings || {};
        el.severity.textContent = (vf.severity_estimate || '--').toUpperCase();
        el.organs.textContent = (vf.organs_involved || []).join(', ') || 'None';
        el.injuryPattern.textContent = vf.injury_pattern || '--';

        el.diffList.innerHTML = '';
        (vf.differential_diagnosis || []).forEach(d => {
            el.diffList.innerHTML += `<li>${d}</li>`;
        });

        el.llmReport.textContent = result.report || '--';
    }

    el.qaSubmit.addEventListener('click', submitQuestion);
    el.qaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') submitQuestion();
    });

    function submitQuestion() {
        if (qaStreaming || !currentSessionId) return;
        const q = el.qaInput.value.trim();
        if (!q) return;

        el.chatHistory.innerHTML += `<div class="message"><div class="msg-body">You: ${q}</div></div>`;
        el.qaInput.value = '';
        qaStreaming = true;

        const aiMsg = document.createElement('div');
        aiMsg.className = 'message ai';
        aiMsg.innerHTML = `<div class="msg-body">AI: <span></span></div>`;
        el.chatHistory.appendChild(aiMsg);
        const span = aiMsg.querySelector('span');

        const evtSource = new EventSource(`/qa-stream?session_id=${currentSessionId}&q=${encodeURIComponent(q)}`);
        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                evtSource.close();
                qaStreaming = false;
                return;
            }
            span.textContent += e.data;
            el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
        };
    }

    el.resetBtn.addEventListener('click', () => location.reload());
});
