'use client';

import { useState, useRef, useEffect } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Send } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

type Role = 'user' | 'assistant';

interface Message {
  id: number;
  role: Role;
  content: string;
}

export default function ChatCPKConversationUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto‑scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Input'a loading bittikten sonra otomatik focus ekle
  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);

  const handleSend = () => {
    // Eğer bir mesaj gönderme işlemi devam ediyorsa, yeni gönderimi engelle
    if (isLoading) return;
    if (!input.trim()) return;

    const newMsg = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
    } as const;

    setMessages((prev) => [...prev, newMsg]);
    setInput(''); // Inputu temizle
    setIsLoading(true); // Loading başlasın

    // 🔧 Stub: Gerçek asistan çağrısıyla değiştir
    const fakeAssistantReply = {
      id: Date.now() + 1,
      role: 'assistant',
      content: '(This is a demo reply)',
    } as const;

    setTimeout(() => {
      setMessages((prev) => [...prev, fakeAssistantReply]);
      setIsLoading(false); // Loading bitsin, useEffect sayesinde input focus olur
    }, 600);
  };

  return (
      <div
          className={cn(
              "flex flex-col justify-between w-full min-h-screen max-h-screen !h-full py-10",
              messages.length === 0 && "!justify-center"
          )}
      >
        {messages.length > 0 && (
            <div className="space-y-3 p-4 h-full overflow-y-auto">
              <AnimatePresence initial={false}>
                {messages.map(({ id, role, content }) => (
                    <motion.div
                        key={id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
                            role === 'user'
                                ? 'ml-auto bg-muted/50 text-text'
                                : 'mr-auto'
                        }`}
                    >
                      {content}
                    </motion.div>
                ))}
              </AnimatePresence>
              <div ref={bottomRef} />
            </div>
        )}

        {messages.length === 0 && (
            <p className="text-center text-4xl font-medium mb-4">
              What can I help with?
            </p>
        )}

        <div className="p-2 rounded-2xl bg-muted/50 w-2/3 mx-auto flex gap-2 items-end">
          <Textarea
              placeholder="Ask anything"
              value={input}
              disabled={isLoading} // Loading sırasında input devre dışı bırak
              ref={inputRef}  // Input ref ekle
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              className="resize-none"
          />
          <Button
              size="icon"
              onClick={handleSend}
              aria-label="Send"
              disabled={isLoading}
          >
            {isLoading ? (
                <div className="animate-spin h-4 w-4 border-2 border-t-muted rounded-full" />
            ) : (
                <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
  );
}
