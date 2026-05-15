// Updated JavaScript for PVO Chatbot - calls backend instead of Anthropic

async function sendMessage(override) {
  if (loading) return;
  const inp = document.getElementById('userInput');
  const text = (override || inp.value).trim();
  if (!text) return;
  inp.value = '';
  addMessage('user', text);
  history.push({ role: 'user', content: text });
  loading = true;
  document.getElementById('sendBtn').disabled = true;
  const wrap = document.getElementById('messages');
  const td = document.createElement('div');
  td.className = 'msg bot';
  td.id = 'typing';
  td.innerHTML = '<div class="avatar">🎖️</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
  wrap.appendChild(td);
  wrap.scrollTop = wrap.scrollHeight;
  try {
    const r = await fetch('http://localhost:8426/api/chat', {  // Change to your backend URL
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history  // Backend adds system prompt internally
      })
    });
    const data = await r.json();
    document.getElementById('typing')?.remove();
    const reply = data.content?.[0]?.text || "Sorry, I could not get a response. Please call (858) 206-8854.";
    history.push({ role: 'assistant', content: reply });
    addMessage('bot', toHtml(reply));
  } catch (e) {
    document.getElementById('typing')?.remove();
    addMessage('bot', 'Sorry, something went wrong. Please <a href="mailto:contact@powayveterans.org">email us</a> or call (858) 206-8854.');
  }
  loading = false;
  document.getElementById('sendBtn').disabled = false;
  inp.focus();
}