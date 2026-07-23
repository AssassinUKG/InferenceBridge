import {
  forwardRef,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: "ib-button-primary",
  secondary: "ib-button-secondary",
  ghost: "ib-button-ghost",
  danger: "ib-button-danger",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-9 px-3.5 text-sm",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      variant = "secondary",
      size = "md",
      icon,
      className = "",
      children,
      type = "button",
      ...props
    },
    ref
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={`ib-button ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
        {...props}
      >
        {icon}
        {children}
      </button>
    );
  }
);

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  size?: "sm" | "md" | "lg";
  selected?: boolean;
  children: ReactNode;
}

const iconSizeClasses = {
  sm: "h-8 w-8",
  md: "h-9 w-9",
  lg: "h-10 w-10",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    {
      label,
      size = "md",
      selected = false,
      className = "",
      children,
      type = "button",
      ...props
    },
    ref
  ) {
    return (
      <span className="ib-tooltip-wrap">
        <button
          ref={ref}
          type={type}
          aria-label={label}
          className={`ib-icon-button ${iconSizeClasses[size]} ${selected ? "is-selected" : ""} ${className}`}
          {...props}
        >
          {children}
        </button>
        <span role="tooltip" className="ib-tooltip">
          {label}
        </span>
      </span>
    );
  }
);
