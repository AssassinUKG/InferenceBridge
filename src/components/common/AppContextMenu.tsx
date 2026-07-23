import {
  Boxes,
  Clipboard,
  Copy,
  Download,
  Gauge,
  Image,
  Link2,
  MessageSquare,
  MessageSquarePlus,
  MousePointer2,
  Redo2,
  Scissors,
  Search,
  Settings,
  Undo2,
} from "lucide-react";
import {
  type ComponentType,
  type SVGProps,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import type { AppNavId } from "./AppSidebar";

type MenuIcon = ComponentType<SVGProps<SVGSVGElement> & { size?: number }>;

interface MenuAction {
  id: string;
  label: string;
  icon: MenuIcon;
  shortcut?: string;
  dividerBefore?: boolean;
  disabled?: boolean;
  run: () => void | Promise<void>;
}

interface MenuState {
  x: number;
  y: number;
  target: HTMLElement;
  selection: string;
}

interface Props {
  activeTab: AppNavId;
  onNavigate: (tab: AppNavId) => void;
  onCreateChat: () => void | Promise<void>;
}

type EditableElement = HTMLInputElement | HTMLTextAreaElement;

function editableElement(target: HTMLElement): EditableElement | HTMLElement | null {
  const candidate = target.closest<HTMLElement>("input, textarea, [contenteditable='true']");
  if (!candidate) return null;
  if (candidate instanceof HTMLInputElement) {
    const textTypes = new Set(["", "text", "search", "url", "tel", "email", "password"]);
    if (!textTypes.has(candidate.type)) return null;
  }
  if (candidate instanceof HTMLInputElement || candidate instanceof HTMLTextAreaElement) {
    if (candidate.disabled || candidate.readOnly) return null;
  }
  return candidate;
}

function editableSelection(element: EditableElement | HTMLElement | null) {
  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    const start = element.selectionStart ?? 0;
    const end = element.selectionEnd ?? start;
    return element.value.slice(start, end);
  }
  return element?.isContentEditable ? window.getSelection()?.toString() ?? "" : "";
}

function setNativeValue(element: EditableElement, value: string) {
  const prototype = element instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype
    : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  setter?.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
}

function replaceSelection(element: EditableElement | HTMLElement, value: string) {
  element.focus();
  if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
    const start = element.selectionStart ?? element.value.length;
    const end = element.selectionEnd ?? start;
    const next = `${element.value.slice(0, start)}${value}${element.value.slice(end)}`;
    setNativeValue(element, next);
    const caret = start + value.length;
    element.setSelectionRange(caret, caret);
    return;
  }
  document.execCommand("insertText", false, value);
}

async function writeClipboard(text: string) {
  if (!text) return;
  await navigator.clipboard.writeText(text);
}

function contextualCopyTarget(target: HTMLElement) {
  return target.closest<HTMLElement>("[data-context-copy]");
}

async function copyImage(source: string) {
  const blob = await fetch(source).then((response) => response.blob());
  await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
}

function saveImage(source: string) {
  const link = document.createElement("a");
  link.href = source;
  link.download = "inferencebridge-image.png";
  link.click();
}

function menuActions(
  state: MenuState,
  activeTab: AppNavId,
  onNavigate: (tab: AppNavId) => void,
  onCreateChat: () => void | Promise<void>,
): MenuAction[] {
  const actions: MenuAction[] = [];
  const editable = editableElement(state.target);
  const editableText = editableSelection(editable);
  const selected = editable ? editableText : state.selection;
  const copyTarget = contextualCopyTarget(state.target);
  const link = state.target.closest<HTMLAnchorElement>("a[href]");
  const image = state.target.closest<HTMLImageElement>("img[src]");

  if (editable) {
    actions.push(
      {
        id: "undo",
        label: "Undo edit",
        icon: Undo2,
        shortcut: "Ctrl+Z",
        run: () => {
          editable.focus();
          document.execCommand("undo");
        },
      },
      {
        id: "redo",
        label: "Redo edit",
        icon: Redo2,
        shortcut: "Ctrl+Y",
        run: () => {
          editable.focus();
          document.execCommand("redo");
        },
      },
      {
        id: "cut",
        label: "Cut",
        icon: Scissors,
        shortcut: "Ctrl+X",
        dividerBefore: true,
        disabled: !selected,
        run: async () => {
          await writeClipboard(selected);
          replaceSelection(editable, "");
        },
      },
      {
        id: "copy",
        label: "Copy",
        icon: Copy,
        shortcut: "Ctrl+C",
        disabled: !selected,
        run: () => writeClipboard(selected),
      },
      {
        id: "paste",
        label: "Paste",
        icon: Clipboard,
        shortcut: "Ctrl+V",
        disabled: typeof navigator.clipboard?.readText !== "function",
        run: async () => replaceSelection(editable, await navigator.clipboard.readText()),
      },
      {
        id: "select-all",
        label: "Select all",
        icon: MousePointer2,
        shortcut: "Ctrl+A",
        run: () => {
          editable.focus();
          if (editable instanceof HTMLInputElement || editable instanceof HTMLTextAreaElement) {
            editable.select();
          } else {
            document.execCommand("selectAll");
          }
        },
      },
    );
  } else if (state.selection) {
    actions.push({
      id: "copy-selection",
      label: "Copy selected text",
      icon: Copy,
      shortcut: "Ctrl+C",
      run: () => writeClipboard(state.selection),
    });
  } else if (copyTarget?.dataset.contextCopy) {
    actions.push({
      id: "copy-context",
      label: `Copy ${copyTarget.dataset.contextLabel || "content"}`,
      icon: Copy,
      run: () => writeClipboard(copyTarget.dataset.contextCopy ?? ""),
    });
  }

  if (image?.src) {
    actions.push(
      {
        id: "copy-image",
        label: "Copy image",
        icon: Image,
        dividerBefore: actions.length > 0,
        disabled: typeof ClipboardItem === "undefined" || typeof navigator.clipboard?.write !== "function",
        run: () => copyImage(image.src),
      },
      {
        id: "save-image",
        label: "Save image",
        icon: Download,
        run: () => saveImage(image.src),
      },
    );
  }

  if (link?.href) {
    actions.push(
      {
        id: "open-link",
        label: "Open link",
        icon: Link2,
        dividerBefore: actions.length > 0,
        run: () => {
          window.open(link.href, "_blank", "noopener,noreferrer");
        },
      },
      {
        id: "copy-link",
        label: "Copy link address",
        icon: Copy,
        run: () => writeClipboard(link.href),
      },
    );
  }

  const navigation: Array<[AppNavId, string, MenuIcon]> = [
    ["chat", "Open Chat", MessageSquare],
    ["models", "Open Models", Boxes],
    ["browse", "Open Model Hub", Search],
    ["benchmark", "Open Benchmarks", Gauge],
    ["settings", "Open Settings", Settings],
  ];
  actions.push({
    id: "new-chat",
    label: "New chat",
    icon: MessageSquarePlus,
    dividerBefore: actions.length > 0,
    shortcut: "Ctrl+N",
    run: onCreateChat,
  });
  navigation.forEach(([tab, label, icon]) => {
    if (tab === activeTab) return;
    actions.push({ id: `navigate-${tab}`, label, icon, run: () => onNavigate(tab) });
  });
  return actions;
}

export function AppContextMenu({ activeTab, onNavigate, onCreateChat }: Props) {
  const [menu, setMenu] = useState<MenuState | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const open = (event: MouseEvent) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (!target || menuRef.current?.contains(target)) return;
      event.preventDefault();
      setMenu({
        x: event.clientX,
        y: event.clientY,
        target,
        selection: window.getSelection()?.toString().trim() ?? "",
      });
    };
    document.addEventListener("contextmenu", open, true);
    return () => document.removeEventListener("contextmenu", open, true);
  }, []);

  useEffect(() => {
    if (!menu) return undefined;
    const close = (event: Event) => {
      if (event instanceof PointerEvent && menuRef.current?.contains(event.target as Node)) return;
      setMenu(null);
    };
    const closeOnKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenu(null);
      }
    };
    window.addEventListener("pointerdown", close, true);
    window.addEventListener("blur", close);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", closeOnKey, true);
    return () => {
      window.removeEventListener("pointerdown", close, true);
      window.removeEventListener("blur", close);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", closeOnKey, true);
    };
  }, [menu]);

  useEffect(() => {
    if (!menu || !menuRef.current) return;
    const rect = menuRef.current.getBoundingClientRect();
    const margin = 8;
    const left = Math.max(margin, Math.min(menu.x, window.innerWidth - rect.width - margin));
    const top = Math.max(margin, Math.min(menu.y, window.innerHeight - rect.height - margin));
    menuRef.current.style.left = `${left}px`;
    menuRef.current.style.top = `${top}px`;
    menuRef.current.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
  }, [menu]);

  if (!menu) return null;
  const actions = menuActions(menu, activeTab, onNavigate, onCreateChat);

  const runAction = (action: MenuAction) => {
    if (action.disabled) return;
    setMenu(null);
    void Promise.resolve(action.run()).catch(() => undefined);
  };

  const handleKeys = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const enabled = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? []);
    const current = enabled.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      enabled[(current + direction + enabled.length) % enabled.length]?.focus();
    } else if (event.key === "Home") {
      event.preventDefault();
      enabled[0]?.focus();
    } else if (event.key === "End") {
      event.preventDefault();
      enabled.at(-1)?.focus();
    }
  };

  return createPortal(
    <div
      ref={menuRef}
      role="menu"
      aria-label="InferenceBridge actions"
      className="ib-context-menu"
      style={{ left: menu.x, top: menu.y }}
      onKeyDown={handleKeys}
    >
      {actions.map((action) => {
        const Icon = action.icon;
        return (
          <div key={action.id} className={action.dividerBefore ? "ib-context-menu-divider" : undefined}>
            <button
              type="button"
              role="menuitem"
              disabled={action.disabled}
              onClick={() => runAction(action)}
            >
              <Icon size={15} aria-hidden="true" />
              <span>{action.label}</span>
              {action.shortcut && <kbd>{action.shortcut}</kbd>}
            </button>
          </div>
        );
      })}
    </div>,
    document.body,
  );
}
