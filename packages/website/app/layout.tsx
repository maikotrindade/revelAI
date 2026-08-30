import type { Metadata, Viewport } from "next";
import "./globals.css";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { ThemeScript } from "@/components/theme-script";

export const metadata: Metadata = {
  title: "RevelAI — split album page scans into individual photos",
  description:
    "Photograph a page of a family album and get each print back, cropped and deskewed with no quality loss. Restoration is a separate stage that never touches your originals.",
  keywords: [
    "photo restoration",
    "album scanning",
    "photo splitting",
    "genealogy",
    "OpenCV",
  ],
  authors: [{ name: "Maiko Trindade" }],
  openGraph: {
    title: "RevelAI",
    description:
      "Split album page scans into individual photos, then restore them with AI.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfaf7" },
    { media: "(prefers-color-scheme: dark)", color: "#111110" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body>
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <SiteHeader />
        <main id="main">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
