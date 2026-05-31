export function parseCliArgs(input: string): string[] {
  const args: string[] = [];
  const pattern = /"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|[^\s,]+/g;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(input)) !== null) {
    const value = (match[1] ?? match[2] ?? match[0])
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'")
      .replace(/\\\\/g, "\\")
      .trim();
    if (value) args.push(value);
  }

  return args;
}

export function formatCliArgs(args: string[] | null | undefined): string {
  return (args ?? []).join(" ");
}
