    const r = await fetch('http://localhost:8426/api/chat', {  // Change to your backend URL
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history  // Backend adds system prompt internally
      })
    });