'use client';

import {useState, useRef, useEffect} from 'react';
import {Textarea} from '@/components/ui/textarea';
import {Button} from '@/components/ui/button';
import {Send} from 'lucide-react';
import {motion, AnimatePresence} from 'framer-motion';
import {cn} from '@/lib/utils';

type Role = 'user' | 'assistant';

interface Message {
    id: number;
    role: Role;
    content: string;
    isLoading?: boolean;  // Add this property
}

const cleanResponse = (text: string) => {
    return text
        .replace(/^{"generated_text":"/, '') // Remove the leading format
        .replace(/"}%$/, '')                 // Remove the trailing format
        .trim();
};

export default function ChatCPKConversationUI() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const bottomRef = useRef<HTMLDivElement | null>(null);
    const inputRef = useRef<HTMLTextAreaElement | null>(null);

    // Auto‑scroll to the latest message
    useEffect(() => {
        bottomRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages]);

    // Input'a loading bittikten sonra otomatik focus ekle
    useEffect(() => {
        if (!isLoading) {
            inputRef.current?.focus();
        }
    }, [isLoading]);

    const handleSend = async () => {
        if (isLoading) return;
        if (!input.trim()) return;

        setError(null);
        const newMsg = {
            id: Date.now(),
            role: 'user',
            content: input.trim(),
        } as const;

        // Add a loading message immediately
        const loadingMsg = {
            id: Date.now() + 1,
            role: 'assistant',
            content: 'AI is thinking...',
            isLoading: true
        };

        setMessages((prev) => [...prev, newMsg, loadingMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: newMsg.content,
                    max_new_tokens: 150,
                    temperature: 0.95,
                    top_p: 0.7,
                    do_sample: true
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data);

            // Replace loading message with actual response
            setMessages((prev) => prev.filter(msg => !msg.isLoading));
            
            const assistantReply = {
                id: Date.now() + 1,
                role: 'assistant',
                content: data.generated_text,
            } as const;

            setMessages((prev) => [...prev, assistantReply]);
        } catch (error) {
            console.error('Error:', error);
            setError(error instanceof Error ? error.message : 'Failed to get response');
            
            // Replace loading message with error message
            setMessages((prev) => prev.filter(msg => !msg.isLoading));
            
            const errorMsg = {
                id: Date.now() + 1,
                role: 'assistant',
                content: 'Sorry, I encountered an error while processing your request. Please try again.',
            } as const;

            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
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
                        {messages.map(({id, role, content, isLoading}) => (
                            <motion.div
                                key={id}
                                initial={{opacity: 0, y: 10}}
                                animate={{opacity: 1, y: 0}}
                                exit={{opacity: 0, y: -10}}
                                transition={{duration: 0.2}}
                                className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
                                    role === 'user'
                                        ? 'ml-auto bg-muted/50 text-text'
                                        : 'mr-auto'
                                } ${isLoading ? 'animate-pulse' : ''}`}
                            >
                                {content}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    <div ref={bottomRef}/>
                </div>
            )}

            {messages.length === 0 && (
                <p className="text-center text-4xl font-medium mb-4">
                    What can I help with?
                </p>
            )}

            {error && (
                <div className="text-red-500 text-sm text-center mb-2">
                    {error}
                </div>
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
                        <div className="animate-spin h-4 w-4 border-2 border-t-muted rounded-full"/>
                    ) : (
                        <Send className="h-4 w-4"/>
                    )}
                </Button>
            </div>
        </div>
    );
}
