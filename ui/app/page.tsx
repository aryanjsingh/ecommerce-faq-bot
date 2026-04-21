"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  ShoppingCart,
  Send,
  RefreshCw,
  Bot,
  User,
  Sparkles,
  Package,
  CreditCard,
  RotateCcw,
  Truck,
} from "lucide-react";

const API_BASE = "http://localhost:8000";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  loading?: boolean;
}

const QUICK_QUESTIONS = [
  { label: "Return Policy", icon: RotateCcw, q: "What is your return policy?" },
  { label: "Shipping Times", icon: Truck, q: "How long does shipping take?" },
  { label: "Payment Methods", icon: CreditCard, q: "What payment methods do you accept?" },
  { label: "Track Order", icon: Package, q: "How do I track my order?" },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isLoading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: question.trim(),
      };
      const loadingMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setInput("");
      setIsLoading(true);

      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question.trim(), thread_id: threadId }),
        });

        if (!res.ok) throw new Error("API error");
        const data = await res.json();

        setThreadId(data.thread_id);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? { ...m, content: data.answer, sources: data.sources, loading: false }
              : m
          )
        );
      } catch {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingMsg.id
              ? {
                  ...m,
                  content:
                    "⚠️ Could not reach the API. Make sure `uvicorn api:app` is running on port 8000.",
                  loading: false,
                }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [threadId, isLoading]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const resetConversation = () => {
    setMessages([]);
    setThreadId(null);
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-card/80 backdrop-blur-md">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
              <ShoppingCart className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h1 className="font-semibold text-foreground text-sm leading-tight">ShopBot</h1>
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-green-500 inline-block" />
                AI Customer Support
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs gap-1">
              <Sparkles className="h-3 w-3" />
              Qwen2.5:3b
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={resetConversation}
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
              title="New conversation"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <ScrollArea className="flex-1">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">

          {/* Welcome state */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 gap-6 text-center">
              <div className="h-16 w-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <ShoppingCart className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-foreground">How can I help you today?</h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Ask me anything about returns, shipping, payments, or your order.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2 w-full max-w-md">
                {QUICK_QUESTIONS.map(({ label, icon: Icon, q }) => (
                  <button
                    key={label}
                    onClick={() => sendMessage(q)}
                    className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-border bg-card text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors text-left"
                  >
                    <Icon className="h-4 w-4 shrink-0 text-primary" />
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <Avatar className="h-8 w-8 shrink-0 mt-0.5">
                <AvatarFallback
                  className={
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground text-xs"
                      : "bg-muted text-muted-foreground text-xs"
                  }
                >
                  {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </AvatarFallback>
              </Avatar>

              <div className={`flex flex-col gap-1 max-w-[80%] ${msg.role === "user" ? "items-end" : ""}`}>
                {msg.loading ? (
                  <div className="rounded-2xl rounded-tl-sm bg-card border border-border px-4 py-3 space-y-2">
                    <Skeleton className="h-3 w-48" />
                    <Skeleton className="h-3 w-36" />
                    <Skeleton className="h-3 w-24" />
                  </div>
                ) : (
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "rounded-tr-sm bg-primary text-primary-foreground"
                        : "rounded-tl-sm bg-card border border-border text-foreground"
                    }`}
                  >
                    {msg.content}
                  </div>
                )}

                {msg.sources && msg.sources.length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1">
                    {msg.sources.map((s) => (
                      <Badge key={s} variant="secondary" className="text-xs py-0">
                        {s}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t border-border bg-card/80 backdrop-blur-md">
        <div className="max-w-3xl mx-auto px-4 py-3">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about returns, shipping, payments..."
              disabled={isLoading}
              className="flex-1 bg-background border-border focus-visible:ring-primary"
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              size="icon"
              className="shrink-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
          <Separator className="my-2" />
          <p className="text-center text-xs text-muted-foreground">
            Powered by Qwen2.5:3b via Ollama • 100% local • No data leaves your machine
          </p>
        </div>
      </div>
    </div>
  );
}
