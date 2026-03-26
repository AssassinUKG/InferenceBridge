interface Props {
  text: string;
}

export function StreamingText({ text }: Props) {
  return (
    <div className="flex gap-3 px-4 py-3">
      <div className="w-7 h-7 rounded-full bg-purple-600/30 flex items-center justify-center text-xs text-purple-300 shrink-0 mt-0.5">
        AI
      </div>
      <div className="flex-1 min-w-0">
        {text ? (
          <p className="text-sm text-gray-300 whitespace-pre-wrap break-words">
            {text}
            <span className="inline-block w-2 h-4 bg-blue-400 animate-pulse ml-0.5 align-text-bottom" />
          </p>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
            <span>Generating...</span>
          </div>
        )}
      </div>
    </div>
  );
}
