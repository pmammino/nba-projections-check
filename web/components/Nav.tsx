"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [["/", "Overview"], ["/paper", "Filtered paper trading"], ["/calibration", "Calibration"]] as const;

export default function Nav() {
  const path = usePathname();
  return (
    <nav>
      <div className="in">
        <span className="brand">NBA paper trading</span>
        {TABS.map(([href, name]) => (
          <Link key={href} href={href} className={path === href ? "on" : ""}>{name}</Link>
        ))}
      </div>
    </nav>
  );
}
