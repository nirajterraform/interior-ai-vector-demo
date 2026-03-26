"use client";
import { useState } from "react";

/**
 * Test page for subject reference inpainting
 * Access at: http://localhost:3000/test-inpaint
 *
 * This lets you test whether Imagen can place your actual
 * catalogue products into a room photo.
 */

// ── Hardcoded test products from your catalogue ───────────────────────────────
// Replace these URLs with real product image URLs from your catalogue
// You can grab these from prhomz.com or from your DB
// Real catalogue products from prhomz.com
// For non-square room photos, only the FIRST product is used (Imagen API limit)
// Test with the sofa first — it's the most impactful
const TEST_PRODUCTS = [
  {
    imageUrl: "https://cdn.shopify.com/s/files/1/0582/1333/5109/files/edb5dcd14b97d1c9af7afe92048a4261.jpg?v=1772475080",
    title: "Beige 105 Inch L Shaped Sectional Sofa With Storage Ottoman",
    category: "sofa",
  },
  {
    imageUrl: "https://cdn.shopify.com/s/files/1/0582/1333/5109/files/glov-23.jpg?v=1772498716",
    title: "Glover Natural Braided Jute Carpet",
    category: "rug",
  },
  {
    imageUrl: "https://cdn.shopify.com/s/files/1/0582/1333/5109/files/EHCRCT1459581-2_b5448e2b-5bd1-4f3a-b4ec-ce99498e5f06.jpg?v=1769019404",
    title: "Modern Industrial Style Solid Wood Coffee Table",
    category: "coffee_table",
  },
];

export default function TestInpaintPage() {
  const [roomImage, setRoomImage]     = useState<string | null>(null);
  const [result, setResult]           = useState<string | null>(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [log, setLog]                 = useState<string>("");
  const [productUrls, setProductUrls] = useState(
    // For non-square room photos, only 1 product is used (Imagen API limit)
    // Start with just the sofa — most impactful to test
    TEST_PRODUCTS[0].imageUrl
  );
  const [theme, setTheme]   = useState("japandi");
  const [roomType, setRoomType] = useState("living_room");

  function handleRoomUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setRoomImage(reader.result as string);
    reader.readAsDataURL(file);
  }

  async function runTest() {
    if (!roomImage) { setError("Please upload a room photo first"); return; }
    setLoading(true); setError(null); setResult(null); setLog("");

    // Parse product URLs — allow custom URLs or use defaults
    const urls = productUrls.split("\n").map((u) => u.trim()).filter(Boolean);
    const products = urls.slice(0, 3).map((url, i) => ({
      imageUrl: url,
      title: TEST_PRODUCTS[i]?.title || `Product ${i + 1}`,
      category: TEST_PRODUCTS[i]?.category || "sofa",
    }));

    setLog(`Testing with ${products.length} products:\n` +
      products.map((p, i) => `  [${i+1}] ${p.category}: ${p.title}`).join("\n")
    );

    try {
      const res = await fetch("/api/inpaint-with-products", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageBase64: roomImage,
          mimeType: roomImage.split(";")[0].split(":")[1],
          theme,
          roomType,
          products,
        }),
      });

      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      setResult(data.generatedImage);
      setLog((prev) => prev + "\n\n✅ Success!\nPrompt used:\n" + data.prompt);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLog((prev) => prev + "\n\n❌ Failed: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif", maxWidth: 1200 }}>
      <h1>🧪 Subject Reference Inpainting Test</h1>
      <p style={{ color: "#666" }}>
        Tests whether Imagen can place your actual catalogue products into a room photo.
        Uses <code>REFERENCE_TYPE_SUBJECT</code> with <code>imagen-3.0-capability-001</code>.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 24 }}>

        {/* ── LEFT: Controls ─────────────────────────────────────────── */}
        <div>
          <h2>1. Upload Room Photo</h2>
          <input type="file" accept="image/*" onChange={handleRoomUpload} />
          {roomImage && (
            <img src={roomImage} alt="room" style={{ width: "100%", marginTop: 8, borderRadius: 8 }} />
          )}

          <h2 style={{ marginTop: 24 }}>2. Theme & Room Type</h2>
          <div style={{ display: "flex", gap: 12 }}>
            <input
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="Theme (e.g. japandi)"
              style={{ flex: 1, padding: 8, border: "1px solid #ddd", borderRadius: 4 }}
            />
            <select
              value={roomType}
              onChange={(e) => setRoomType(e.target.value)}
              style={{ padding: 8, border: "1px solid #ddd", borderRadius: 4 }}
            >
              <option value="living_room">Living Room</option>
              <option value="bedroom">Bedroom</option>
              <option value="dining_room">Dining Room</option>
            </select>
          </div>

          <h2 style={{ marginTop: 24 }}>3. Product Image URLs (max 3, one per line)</h2>
          <p style={{ color: "#888", fontSize: 13 }}>
            Paste Shopify CDN URLs of your catalogue products. These are the actual images
            that Imagen will try to place into the room.
          </p>
          <textarea
            value={productUrls}
            onChange={(e) => setProductUrls(e.target.value)}
            rows={6}
            style={{ width: "100%", padding: 8, fontFamily: "monospace", fontSize: 12, border: "1px solid #ddd", borderRadius: 4, boxSizing: "border-box" }}
          />
          <p style={{ color: "#888", fontSize: 12 }}>
            Product categories assumed (in order): sofa, rug, coffee_table
          </p>

          <button
            onClick={runTest}
            disabled={loading || !roomImage}
            style={{
              marginTop: 16, padding: "12px 24px", background: loading ? "#aaa" : "#1a1a1a",
              color: "white", border: "none", borderRadius: 8, cursor: loading ? "not-allowed" : "pointer",
              fontSize: 16, width: "100%",
            }}
          >
            {loading ? "⏳ Generating... (30-60 seconds)" : "🚀 Test Subject Reference Inpainting"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: 12, background: "#fff0f0", border: "1px solid #ffcccc", borderRadius: 8, color: "#cc0000" }}>
              ❌ {error}
            </div>
          )}

          {log && (
            <pre style={{ marginTop: 12, padding: 12, background: "#f5f5f5", borderRadius: 8, fontSize: 12, overflow: "auto", whiteSpace: "pre-wrap" }}>
              {log}
            </pre>
          )}
        </div>

        {/* ── RIGHT: Results ──────────────────────────────────────────── */}
        <div>
          <h2>Result</h2>
          {result ? (
            <div>
              <img src={result} alt="generated" style={{ width: "100%", borderRadius: 8 }} />
              <a
                href={result}
                download="inpaint-result.png"
                style={{ display: "block", marginTop: 8, textAlign: "center", color: "#0066cc" }}
              >
                Download result
              </a>
            </div>
          ) : (
            <div style={{ height: 400, background: "#f0f0f0", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#999" }}>
              Result will appear here
            </div>
          )}

          {/* Preview the product images being used */}
          <h3 style={{ marginTop: 24 }}>Product References Being Used</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            {productUrls.split("\n").filter(Boolean).slice(0, 3).map((url, i) => (
              <div key={i} style={{ textAlign: "center" }}>
                <img
                  src={url}
                  alt={`product ${i+1}`}
                  style={{ width: "100%", height: 120, objectFit: "contain", background: "#f5f5f5", borderRadius: 4 }}
                  onError={(e) => { (e.target as HTMLImageElement).style.opacity = "0.3"; }}
                />
                <p style={{ fontSize: 11, color: "#666", margin: "4px 0" }}>
                  [{i+1}] {TEST_PRODUCTS[i]?.category || "product"}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
