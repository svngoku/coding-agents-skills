import {
  Blocks,
  Bot,
  Cloud,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  Layers3,
  PanelsTopLeft,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";

const CATEGORY_ICONS: Record<string, LucideIcon> = {
  "AI & Agents": Bot,
  "Architecture & Design": Blocks,
  "Data & Storage": Database,
  "Testing & Quality": FlaskConical,
  Security: ShieldCheck,
  Performance: Gauge,
  "DevOps & Cloud": Cloud,
  "UI & Frontend": PanelsTopLeft,
  "Post-Training & ML": Cpu,
};

export function getCategoryIcon(category: string): LucideIcon {
  return CATEGORY_ICONS[category] ?? Layers3;
}
