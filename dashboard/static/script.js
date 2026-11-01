// ── TradeAgent Dashboard Script ──
const fmt = (v) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(v);

let countdown = 5;

// Countdown timer UI
setInterval(() => {
    countdown--;
    if (countdown <= 0) countdown = 5;
    const el = document.getElementById('refresh-timer');
    if (el) el.innerText = `Refreshing in ${countdown}s`;
}, 1000);

// ── Account ──
const updateAccount = async () => {
    try {
        const data = await fetch('/api/account').then(r => r.json());
        document.getElementById('acc-balance').innerText = fmt(data.balance || 0);
        document.getElementById('acc-equity').innerText = fmt(data.equity || 0);
        document.getElementById('acc-margin').innerText = fmt(data.margin || 0);
        document.getElementById('acc-free').innerText = fmt(data.margin_free || 0);
        document.getElementById('acc-leverage').innerText = `1:${data.leverage || 1}`;
    } catch (e) { console.error('Account error:', e); }
};

// ── Positions ──
const updatePositions = async () => {
    try {
        const data = await fetch('/api/positions').then(r => r.json());
        const tbody = document.getElementById('positions-body');
        const badge = document.getElementById('pos-count');

        const positions = data.positions || [];
        badge.innerText = `${positions.length} Active`;

        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:2rem">No active positions or orders</td></tr>';
            return;
        }

        tbody.innerHTML = positions.map(pos => {
            const typeClass = pos.is_pending ? 'type-pending' : (pos.type === 'BUY' ? 'type-buy' : 'type-sell');
            const profitClass = pos.profit >= 0 ? 'profit-pos' : 'profit-neg';
            const profitText = pos.is_pending ? '—' : fmt(pos.profit);
            return `<tr>
                <td><strong>${pos.symbol}</strong><br><span style="font-size:0.7rem;color:var(--text-dim)">#${pos.ticket}</span></td>
                <td class="${typeClass}">${pos.type}</td>
                <td>${pos.volume.toFixed(2)}</td>
                <td>${pos.open_price.toFixed(2)}</td>
                <td>${pos.current_price.toFixed(2)}</td>
                <td class="${profitClass}">${profitText}</td>
            </tr>`;
        }).join('');
    } catch (e) { console.error('Positions error:', e); }
};

// ── Sentiment ──
const updateSentiment = async () => {
    try {
        const data = await fetch('/api/sentiment').then(r => r.json());
        const circle = document.getElementById('sentiment-circle');
        const toneEl = document.getElementById('sentiment-tone');

        const scoreStr = data.score > 0 ? `+${data.score.toFixed(2)}` : data.score.toFixed(2);
        document.getElementById('sentiment-score').innerText = scoreStr;
        toneEl.innerText = data.tone;
        document.getElementById('sentiment-time').innerText = `Updated: ${data.last_update}`;

        circle.className = 'sentiment-score-circle';
        toneEl.className = 'tone';
        if (data.tone === 'BULLISH') { circle.classList.add('bullish'); toneEl.classList.add('bullish'); }
        else if (data.tone === 'BEARISH') { circle.classList.add('bearish'); toneEl.classList.add('bearish'); }
    } catch (e) { console.error('Sentiment error:', e); }
};

// ── Strategies ──
const updateStrategies = async () => {
    try {
        const data = await fetch('/api/strategies').then(r => r.json());
        const container = document.getElementById('strategy-container');

        if (!data.strategies || data.strategies.length === 0) {
            container.innerHTML = '<div style="color:var(--text-dim)">No optimized strategies loaded yet.</div>';
            return;
        }

        container.innerHTML = data.strategies.map(strat => {
            const params = strat.parameters || strat.best_params || {};
            const paramsHtml = Object.entries(params).map(([k, v]) =>
                `<div class="param-row">
                    <span class="p-name">${k}</span>
                    <span class="p-val">${typeof v === 'number' ? v.toFixed(4) : v}</span>
                </div>`
            ).join('');

            const wr = strat.win_rate_pct != null ? strat.win_rate_pct.toFixed(1) : '--';
            const name = strat.strategy_name || 'Strategy';

            return `<div class="strategy-card">
                <div class="strat-header">
                    <div>
                        <div class="strat-name">${name}</div>
                        <div class="strat-wr-label">AI-Optimized Parameters</div>
                    </div>
                    <div class="win-rate-badge">WR: ${wr}%</div>
                </div>
                <div class="strat-params">${paramsHtml}</div>
            </div>`;
        }).join('');
    } catch (e) { console.error('Strategies error:', e); }
};

// ── Bot Log ──
const updateBotLog = async () => {
    try {
        const data = await fetch('/api/bot_status').then(r => r.json());
        const container = document.getElementById('bot-log');

        const html = (data.log_lines || []).map(line => {
            let cls = 'log-line info';
            if (line.includes('APPROVED')) cls = 'log-line approved';
            else if (line.includes('WARNING') || line.includes('Rejecting') || line.includes('PAUSED')) cls = 'log-line rejected';
            else if (line.includes('ERROR') || line.includes('BREACHED') || line.includes('Halting')) cls = 'log-line warning';
            return `<span class="${cls}">${escapeHtml(line)}</span>`;
        }).join('\n');

        container.innerHTML = html || '<span style="color:var(--text-dim)">No log data yet...</span>';
        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    } catch (e) { console.error('Log error:', e); }
};

function escapeHtml(text) {
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Initial Load + Polling ──
const refreshAll = () => {
    updateAccount();
    updatePositions();
    updateSentiment();
    updateStrategies();
    updateBotLog();
};

refreshAll();
setInterval(refreshAll, 5000);
