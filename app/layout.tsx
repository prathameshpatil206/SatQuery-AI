import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "GeoAgent VQA | Agentic Remote Sensing Intelligence",
  description: "High-performance split-screen dashboard for Agentic Remote Sensing Visual Question Answering (VQA) with live Leaflet spatial mapping and execution traces.",
  keywords: ["Remote Sensing", "VQA", "GeoAI", "Satellite Imagery", "Leaflet", "Karnataka", "Hubli"],
};

export const viewport: Viewport = {
  themeColor: "#020617",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="bg-slate-950 text-slate-100 antialiased overflow-hidden min-h-screen">
        {children}
      </body>
    </html>
  );
}
