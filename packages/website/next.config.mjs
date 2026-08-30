/** @type {import('next').NextConfig} */
const nextConfig = {
  // A static export, and this is a product decision rather than a deployment
  // one. RevelAI's promise is that family photographs stay on the owner's
  // machine; a site that could accept an upload would contradict the thing the
  // product is selling. Static export means there is nowhere to send one.
  output: "export",

  // GitHub Pages serves a project site from /<repo>, so asset paths need the
  // prefix in production but not in development.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
  images: { unoptimized: true },

  // Directory-style URLs, so /honest-limitations works without a server.
  trailingSlash: true,
  reactStrictMode: true,
};

export default nextConfig;
