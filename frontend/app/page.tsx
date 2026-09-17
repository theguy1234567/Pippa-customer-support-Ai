"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { analyzeMessage, getHealth } from "@/lib/api";
import type { ConversationTurn, HealthResponse, SupportResponse } from "@/lib/types";

const readable = (value: string) => value.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());

type ChatItem = { role: "user" | "assistant"; content: string; result?: SupportResponse };

export default function Home() {
  const [message, setMessage] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [chat, setChat] = useState<ChatItem[]>([]);
  const [result, setResult] = useState<SupportResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { getHealth().then(setHealth).catch(() => setHealth(null)); }, []);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [chat, loading, error]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const current = message.trim();
    if (!current || loading || !health) return;
    setMessage(""); setError(""); setCopied(false); setLoading(true);
    const history: ConversationTurn[] = chat.map(({ role, content }) => ({ role, content }));
    try {
      const response = await analyzeMessage(current, history);
      setResult(response);
      setChat((previous) => [...previous, { role: "user", content: current }, { role: "assistant", content: response.reply ?? "", result: response }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Pippa could not reach the support service.");
    } finally { setLoading(false); }
  }

  function clearChat() { setChat([]); setResult(null); setError(""); setMessage(""); }
  const reply = result?.reply ?? result?.response;

  async function copyReply() { if (!reply) return; await navigator.clipboard.writeText(reply); setCopied(true); }

  return <main className="min-h-screen bg-[#f7f7fb] text-slate-900">
    <header className="border-b border-slate-200 bg-white px-4 py-4 sm:px-8"><div className="mx-auto flex max-w-7xl items-center justify-between"><div><h1 className="font-bold tracking-tight">Pippa Support</h1><p className="text-xs text-slate-500">Evidence-grounded AppleSupport copilot</p></div><div className="flex items-center gap-3"><span className={`rounded-full px-3 py-1.5 text-xs font-medium ${health ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"}`}>{health ? "Service connected" : "Service unavailable"}</span><button onClick={clearChat} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium hover:bg-slate-50">Clear chat</button></div></div></header>
    <div className="mx-auto grid max-w-7xl gap-6 p-4 sm:p-8 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section className="flex min-h-[650px] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-100 px-5 py-4"><p className="text-sm font-semibold">Customer conversation</p><p className="text-xs text-slate-500">Pippa keeps user context across follow-up questions.</p></div>
        <div className="flex-1 space-y-4 overflow-y-auto bg-gradient-to-b from-white to-violet-50/30 p-5">
          {chat.length === 0 && !loading && <div className="mx-auto mt-24 max-w-sm text-center"><div className="text-4xl">🐝</div><h2 className="mt-4 font-semibold">Hi, I’m Pippa.</h2><p className="mt-2 text-sm leading-6 text-slate-500">Ask a customer-support question and I’ll use real AppleSupport history to ground the reply.</p></div>}
          {chat.map((item, index) => item.role === "user" ? <div key={index} className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-violet-600 px-4 py-3 text-sm leading-6 text-white">{item.content}</div> : <div key={index} className="flex max-w-[85%] gap-3"><div className="text-lg">🐝</div><div className="whitespace-pre-wrap rounded-2xl rounded-bl-md bg-slate-100 px-4 py-3 text-sm leading-6 text-slate-800">{item.content || "No grounded answer was available."}</div></div>)}
          {loading && <div className="text-sm text-slate-500">Pippa is checking relevant historical cases…</div>}
          {error && <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{error}</div>}
          <div ref={chatEnd} />
        </div>
        <form onSubmit={submit} className="border-t border-slate-200 p-4"><textarea value={message} onChange={(event) => setMessage(event.target.value)} maxLength={4000} rows={3} placeholder="Type the customer’s message…" className="block w-full resize-none rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none focus:border-violet-500" /><div className="mt-2 flex justify-end"><button type="submit" disabled={loading || !health || !message.trim()} className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{loading ? "Thinking…" : "Ask Pippa"}</button></div></form>
      </section>
      <aside className="space-y-5">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="text-sm font-bold">AI analysis</h2>{result ? <div className="mt-4 space-y-4 text-sm"><div><p className="text-xs font-medium uppercase tracking-wide text-slate-400">Intent</p><p className="mt-1 font-semibold">{readable(result.intent.name)}</p><p className="text-xs text-slate-500">{(result.intent.confidence * 100).toFixed(1)}% · {result.intent.method}</p></div><div><p className="text-xs font-medium uppercase tracking-wide text-slate-400">Decision</p><p className="mt-1 font-semibold">{readable(result.decision.action)}</p><p className="mt-1 text-xs leading-5 text-slate-500">{result.decision.reason}</p></div><div><p className="text-xs font-medium uppercase tracking-wide text-slate-400">Response</p><p className="mt-1 text-slate-600">{result.grounded ? "Grounded" : "Escalated safely"} · {result.generation_method}</p>{result.error && <p className="mt-2 rounded-lg bg-amber-50 p-2 text-xs leading-5 text-amber-800">{result.error}</p>}{reply && result.grounded && <button onClick={copyReply} className="mt-2 text-xs font-medium text-violet-700">{copied ? "Copied" : "Copy reply"}</button>}</div></div> : <p className="mt-3 text-sm text-slate-500">Analysis appears after a message is processed.</p>}</section>
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="text-sm font-bold">Historical evidence</h2>{result?.evidence?.length ? <div className="mt-4 space-y-3">{result.evidence.map((item) => <details key={`${item.case_id}-${item.tweet_id}`} className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-xs font-medium">Relevant AppleSupport case · {item.similarity.toFixed(2)}</summary><div className="mt-3 space-y-2 border-t border-slate-100 pt-3 text-xs leading-5 text-slate-600"><p><b>Customer:</b> {item.customer_message}</p><p><b>AppleSupport:</b> {item.support_response}</p><p className="text-slate-400">Tweet {item.tweet_id} · {item.source}</p></div></details>)}</div> : <p className="mt-3 text-sm text-slate-500">No sufficiently relevant evidence was selected.</p>}</section>
        {health && <section className="rounded-2xl bg-slate-900 p-5 text-slate-100"><p className="text-sm font-semibold">Service details</p><div className="mt-3 space-y-1 text-xs text-slate-300"><p>Classifier: ready</p><p>Retriever: {health.retriever_loaded ? "ready" : "unavailable"}</p><p>Hugging Face LLM: {health.llm_available ? "configured" : "not configured"}</p></div></section>}
      </aside>
    </div>
  </main>;
}
