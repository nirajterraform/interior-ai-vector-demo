"use client";

import { useMemo, useState } from "react";

type RoomType =
  | "living_room"
  | "bedroom"
  | "dining_room"
  | "kitchen"
  | "office";

type RetrievedProduct = {
  bucket: string;
  product_handle: string;
  title: string;
  category: string | null;
  subcategory: string | null;
  normalized_category: string | null;
  image_url: string | null;
  min_price: number | null;
  max_price: number | null;
  similarity: number;
  pinned?: boolean;
  source?: "catalog" | "innovative";
};

type RetrievalResponse = {
  roomType: RoomType;
  theme: string;
  shortlist: RetrievedProduct[];
  nextRotationCursor: number;
};

type GenerateRoomResponse = {
  ok: true;
  generatedImage: string;
  pinnedProducts?: RetrievedProduct[];
  validationPassed?: boolean;
};

const ROOM_TYPE_OPTIONS: Array<{ value: RoomType; label: string }> = [
  { value: "living_room", label: "Living Room" },
  { value: "bedroom", label: "Bedroom" },
  { value: "dining_room", label: "Dining Room" },
  { value: "kitchen", label: "Kitchen" },
  { value: "office", label: "Office" },
];

function formatRoomTypeLabel(roomType: RoomType | null) {
  if (!roomType) return "";
  return ROOM_TYPE_OPTIONS.find((r) => r.value === roomType)?.label ?? roomType;
}

function groupByBucket(items: RetrievedProduct[]) {
  return items.reduce<Record<string, RetrievedProduct[]>>((acc, item) => {
    if (!acc[item.bucket]) acc[item.bucket] = [];
    acc[item.bucket].push(item);
    return acc;
  }, {});
}

export default function Page() {
  const [roomType, setRoomType] = useState<RoomType | null>(null);
  const [theme, setTheme] = useState("modern scandinavian living room");

  const [uploadedRoomImage, setUploadedRoomImage] = useState<string | null>(null);
  const [cleanedRoomImage, setCleanedRoomImage] = useState<string | null>(null);
  const [generatedRoomImage, setGeneratedRoomImage] = useState<string | null>(null);

  const [retrievalResult, setRetrievalResult] = useState<RetrievalResponse | null>(null);
  const [displayedProducts, setDisplayedProducts] = useState<RetrievedProduct[]>([]);
  const [pinnedProducts, setPinnedProducts] = useState<RetrievedProduct[]>([]);

  const [seenHandles, setSeenHandles] = useState<string[]>([]);
  const [rotationCursor, setRotationCursor] = useState(0);
  const [brokenImageHandles, setBrokenImageHandles] = useState<string[]>([]);

  const [uploading, setUploading] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [retrieving, setRetrieving] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);

  const [cleaningStatus, setCleaningStatus] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<string | null>(null);

  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const visiblePinnedProducts = useMemo(
    () => pinnedProducts.filter((p) => !brokenImageHandles.includes(p.product_handle)),
    [pinnedProducts, brokenImageHandles]
  );

  const visibleDisplayedProducts = useMemo(
    () => displayedProducts.filter((p) => !brokenImageHandles.includes(p.product_handle)),
    [displayedProducts, brokenImageHandles]
  );

  const groupedPinned = useMemo(() => groupByBucket(visiblePinnedProducts), [visiblePinnedProducts]);
  const groupedShortlist = useMemo(() => groupByBucket(visibleDisplayedProducts), [visibleDisplayedProducts]);

  async function fileToDataUrl(file: File): Promise<string> {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function onRoomUpload(file: File | null) {
    if (!file) return;

    try {
      setError(null);
      setWarning(null);
      setUploading(true);

      const dataUrl = await fileToDataUrl(file);

      setUploadedRoomImage(dataUrl);
      setCleanedRoomImage(null);
      setGeneratedRoomImage(null);

      setRetrievalResult(null);
      setDisplayedProducts([]);
      setPinnedProducts([]);

      setSeenHandles([]);
      setRotationCursor(0);
      setBrokenImageHandles([]);

      setCleaningStatus(null);
      setPipelineStatus(null);
    } catch (err) {
      console.error(err);
      setError("Failed to read the uploaded image.");
    } finally {
      setUploading(false);
    }
  }

  function markBrokenImage(handle: string) {
    setBrokenImageHandles((prev) => (prev.includes(handle) ? prev : [...prev, handle]));
  }

  async function runCleanRoom(imageBase64: string) {
    const mimeType =
      imageBase64.startsWith("data:image/jpeg") ? "image/jpeg" : "image/png";

    const res = await fetch("/api/clean-room", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        roomImageBase64: imageBase64,
        mimeType,
      }),
    });

    const data = await res.json();

    if (!res.ok || !data.ok) {
      throw new Error(data?.error || "Failed to clean room.");
    }

    return data as {
      ok: true;
      cleanedImage: string;
      retryUsed?: boolean;
      validationPassed?: boolean;
    };
  }

  async function runRetrieveCatalogue(params: {
    roomType: RoomType;
    theme: string;
    seenHandles?: string[];
    rotationCursor?: number;
    pageSize?: number;
  }) {
    const res = await fetch("/api/retrieve-catalogue", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        roomType: params.roomType,
        theme: params.theme,
        seenHandles: params.seenHandles ?? [],
        rotationCursor: params.rotationCursor ?? 0,
        pageSize: params.pageSize ?? 18,
      }),
    });

    const data = (await res.json()) as RetrievalResponse | { error?: string };

    if (!res.ok) {
      throw new Error(
        "error" in data
          ? data.error || "Failed to retrieve catalogue."
          : "Failed to retrieve catalogue."
      );
    }

    return data as RetrievalResponse;
  }

  async function runGenerateRoom(params: {
    roomType: RoomType;
    theme: string;
    originalRoomBase64: string;
    cleanedRoomBase64: string;
    shortlist: RetrievedProduct[];
    editMode?: boolean;
    baseGeneratedImage?: string | null;
  }) {
    const mimeType =
      params.cleanedRoomBase64.startsWith("data:image/jpeg")
        ? "image/jpeg"
        : "image/png";

    const res = await fetch("/api/generate-room", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        roomType: params.roomType,
        theme: params.theme,
        originalRoomBase64: params.originalRoomBase64,
        cleanedRoomBase64: params.cleanedRoomBase64,
        mimeType,
        shortlist: params.shortlist,
        editMode: params.editMode ?? false,
        baseGeneratedImage: params.baseGeneratedImage ?? null,
      }),
    });

    const data = await res.json();

    if (!res.ok || !data.ok) {
      throw new Error(data?.error || "Failed to generate room.");
    }

    return data as GenerateRoomResponse;
  }

  async function handleCleanRoom() {
    if (!uploadedRoomImage) {
      setError("Please upload a room image first.");
      return;
    }

    let stillWorkingTimer: ReturnType<typeof setTimeout> | null = null;
    let retryHintTimer: ReturnType<typeof setTimeout> | null = null;

    try {
      setError(null);
      setWarning(null);
      setCleaning(true);
      setCleaningStatus("Cleaning room...");

      stillWorkingTimer = setTimeout(() => {
        setCleaningStatus("Still working on cleaning...");
      }, 4000);

      retryHintTimer = setTimeout(() => {
        setCleaningStatus("Retrying clean room for a better result...");
      }, 9000);

      const cleanData = await runCleanRoom(uploadedRoomImage);
      setCleanedRoomImage(cleanData.cleanedImage);

      if (cleanData.retryUsed) {
        setCleaningStatus("Retry completed. Best clean-room result applied.");
      } else {
        setCleaningStatus("Clean room completed.");
      }

      setTimeout(() => {
        setCleaningStatus(null);
      }, 2500);
    } catch (err) {
      console.error(err);
      setCleaningStatus(null);
      setError(err instanceof Error ? err.message : "Failed to clean room.");
    } finally {
      if (stillWorkingTimer) clearTimeout(stillWorkingTimer);
      if (retryHintTimer) clearTimeout(retryHintTimer);
      setCleaning(false);
    }
  }

  async function handleRetrieveCatalogue() {
    if (!roomType) {
      setError("Please select a room type.");
      return;
    }

    if (!theme.trim()) {
      setError("Please enter your design requirement.");
      return;
    }

    try {
      setError(null);
      setWarning(null);
      setRetrieving(true);

      const retrieval = await runRetrieveCatalogue({
        roomType,
        theme,
        seenHandles: [],
        rotationCursor: 0,
        pageSize: 18,
      });

      setRetrievalResult(retrieval);
      setDisplayedProducts(retrieval.shortlist);
      setSeenHandles(retrieval.shortlist.map((x) => x.product_handle));
      setRotationCursor(retrieval.nextRotationCursor);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to retrieve catalogue.");
    } finally {
      setRetrieving(false);
    }
  }

  async function handleLoadMoreProducts() {
    if (!roomType || !theme.trim() || !generatedRoomImage) return;

    try {
      setError(null);
      setLoadingMore(true);

      const retrieval = await runRetrieveCatalogue({
        roomType,
        theme,
        seenHandles,
        rotationCursor,
        pageSize: 18,
      });

      setRetrievalResult(retrieval);
      setDisplayedProducts(retrieval.shortlist);

      setSeenHandles((prev) => [
        ...prev,
        ...retrieval.shortlist
          .map((x) => x.product_handle)
          .filter((h) => !prev.includes(h)),
      ]);

      setRotationCursor(retrieval.nextRotationCursor);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to load more products.");
    } finally {
      setLoadingMore(false);
    }
  }

  async function handleGeneratePipeline() {
    if (!uploadedRoomImage) {
      setError("Please upload a room image first.");
      return;
    }

    if (!roomType) {
      setError("Please select a room type.");
      return;
    }

    if (!theme.trim()) {
      setError("Please enter your design requirement.");
      return;
    }

    try {
      setError(null);
      setWarning(null);
      setPipelineRunning(true);

      let effectiveCleanedImage = cleanedRoomImage;

      if (!effectiveCleanedImage) {
        setPipelineStatus("Cleaning room...");
        const cleanData = await runCleanRoom(uploadedRoomImage);
        effectiveCleanedImage = cleanData.cleanedImage;
        setCleanedRoomImage(cleanData.cleanedImage);
      }

      let effectiveRetrieval = retrievalResult;

      const retrievalMatchesCurrentRequest =
        effectiveRetrieval &&
        effectiveRetrieval.roomType === roomType &&
        effectiveRetrieval.theme === theme &&
        effectiveRetrieval.shortlist?.length > 0;

      if (!retrievalMatchesCurrentRequest) {
        setPipelineStatus("Retrieving catalog products...");
        effectiveRetrieval = await runRetrieveCatalogue({
          roomType,
          theme,
          seenHandles: [],
          rotationCursor: 0,
          pageSize: 18,
        });
        setRetrievalResult(effectiveRetrieval);
        setDisplayedProducts(effectiveRetrieval.shortlist);
        setSeenHandles(effectiveRetrieval.shortlist.map((x) => x.product_handle));
        setRotationCursor(effectiveRetrieval.nextRotationCursor);
      }

      if (!effectiveRetrieval?.shortlist?.length) {
        throw new Error("No catalogue products were retrieved.");
      }

      setPipelineStatus(
        generatedRoomImage ? "Applying design changes..." : "Generating final AI room..."
      );

      const generateData = await runGenerateRoom({
        roomType,
        theme,
        originalRoomBase64: uploadedRoomImage,
        cleanedRoomBase64: effectiveCleanedImage!,
        shortlist: effectiveRetrieval.shortlist,
        editMode: !!generatedRoomImage,
        baseGeneratedImage: generatedRoomImage,
      });

      setGeneratedRoomImage(generateData.generatedImage);

      if (generateData.pinnedProducts?.length) {
        setPinnedProducts(
          generateData.pinnedProducts.map((x) => ({
            ...x,
            pinned: true,
          }))
        );
      }

      setPipelineStatus("Done.");
      setTimeout(() => {
        setPipelineStatus(null);
      }, 2500);
    } catch (err) {
      console.error(err);
      setPipelineStatus(null);
      setError(
        err instanceof Error ? err.message : "Failed to generate the AI room."
      );
    } finally {
      setPipelineRunning(false);
    }
  }

  const canClean = !!uploadedRoomImage && !cleaning && !pipelineRunning;
  const canRetrieve =
    !!roomType && !!theme.trim() && !retrieving && !pipelineRunning;
  const canGenerate =
    !!uploadedRoomImage && !!roomType && !!theme.trim() && !pipelineRunning;

  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900">
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">
            Interior AI Vector Demo
          </h1>
          <p className="mt-2 text-sm text-neutral-600">
            No products are shown until the generated room is ready.
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {warning && (
          <div className="mb-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
            {warning}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
          <section className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-semibold">1. Upload Room</h2>

            <div className="mt-4">
              <label className="block text-sm font-medium text-neutral-700">
                Room image
              </label>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                className="mt-2 block w-full rounded-xl border border-neutral-300 bg-white px-3 py-2 text-sm"
                onChange={(e) => void onRoomUpload(e.target.files?.[0] ?? null)}
              />
              <p className="mt-2 text-xs text-neutral-500">
                Supported: PNG, JPG, JPEG, WEBP
              </p>
              {uploading && (
                <p className="mt-2 text-xs text-neutral-500">Reading image...</p>
              )}
            </div>

            <div className="mt-6">
              <h2 className="text-lg font-semibold">2. Select Room Type</h2>
              <div className="mt-4 grid grid-cols-2 gap-3">
                {ROOM_TYPE_OPTIONS.map((option) => {
                  const selected = roomType === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setRoomType(option.value)}
                      className={[
                        "rounded-2xl border px-4 py-3 text-sm font-medium transition",
                        selected
                          ? "border-neutral-900 bg-neutral-900 text-white"
                          : "border-neutral-300 bg-white text-neutral-800 hover:border-neutral-500",
                      ].join(" ")}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-6">
              <h2 className="text-lg font-semibold">3. Design Requirement</h2>
              <p className="mt-1 text-sm text-neutral-600">
                Example: modern scandinavian living room, change sofa to blue velvet, make it more luxury
              </p>

              <textarea
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
                rows={4}
                className="mt-3 w-full rounded-2xl border border-neutral-300 px-4 py-3 text-sm outline-none placeholder:text-neutral-400 focus:border-neutral-500"
                placeholder="Describe the desired room style or change request..."
              />
            </div>

            <div className="mt-6 flex flex-col gap-3">
              <button
                type="button"
                onClick={() => void handleCleanRoom()}
                disabled={!canClean}
                className="rounded-2xl bg-neutral-900 px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                {cleaning ? "Cleaning room..." : "Clean Room"}
              </button>

              {cleaningStatus && (
                <p className="text-sm text-neutral-600">{cleaningStatus}</p>
              )}

              <button
                type="button"
                onClick={() => void handleRetrieveCatalogue()}
                disabled={!canRetrieve}
                className="rounded-2xl border border-neutral-900 bg-white px-4 py-3 text-sm font-medium text-neutral-900 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {retrieving ? "Retrieving products..." : "Retrieve Products"}
              </button>

              <button
                type="button"
                onClick={() => void handleGeneratePipeline()}
                disabled={!canGenerate}
                className="rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                {pipelineRunning
                  ? generatedRoomImage
                    ? "Applying changes..."
                    : "Working..."
                  : generatedRoomImage
                    ? "Apply Changes"
                    : "Generate AI Room"}
              </button>

              {pipelineStatus && (
                <p className="text-sm text-neutral-600">{pipelineStatus}</p>
              )}
            </div>

            <div className="mt-6 rounded-2xl border border-neutral-200 bg-neutral-50 p-4 text-sm text-neutral-700">
              <div>
                <span className="font-medium">Selected Room:</span>{" "}
                {roomType ? formatRoomTypeLabel(roomType) : "Not selected"}
              </div>
              <div className="mt-2">
                <span className="font-medium">Uploaded Image:</span>{" "}
                {uploadedRoomImage ? "Available" : "Not uploaded"}
              </div>
              <div className="mt-2">
                <span className="font-medium">Pinned Products:</span>{" "}
                {visiblePinnedProducts.length}
              </div>
              <div className="mt-2">
                <span className="font-medium">Current Product Page:</span>{" "}
                {visibleDisplayedProducts.length}
              </div>
              <div className="mt-2">
                <span className="font-medium">Generated Room:</span>{" "}
                {generatedRoomImage ? "Available" : "Not generated yet"}
              </div>
            </div>
          </section>

          <section className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <ImageCard
                title="Uploaded Room"
                image={uploadedRoomImage}
                emptyText="No room uploaded yet"
              />
              <ImageCard
                title="Generated Room"
                image={generatedRoomImage}
                emptyText="Generated room will appear here"
              />
            </div>

            <div className="rounded-3xl border border-neutral-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold">Recommended Products</h2>
                  <p className="mt-1 text-sm text-neutral-600">
                    Pinned products stay at the top. Load More changes only the current page around them.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => void handleLoadMoreProducts()}
                  disabled={!generatedRoomImage || !roomType || !theme.trim() || loadingMore}
                  className="rounded-2xl border border-neutral-900 bg-white px-4 py-2 text-sm font-medium text-neutral-900 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loadingMore ? "Loading..." : "Load More Products"}
                </button>
              </div>

              {!generatedRoomImage ? (
                <div className="mt-6 rounded-2xl border border-dashed border-neutral-300 bg-neutral-50 px-4 py-10 text-center text-sm text-neutral-500">
                  Generate AI Room first to see pinned products and recommended products.
                </div>
              ) : (
                <div className="mt-6 space-y-6">
                  {visiblePinnedProducts.length > 0 && (
                    <div className="space-y-4">
                      <div className="flex items-center gap-3">
                        <h3 className="text-sm font-semibold uppercase tracking-wide text-neutral-700">
                          In your room
                        </h3>
                        <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">
                          Pinned
                        </span>
                      </div>

                      {Object.entries(groupedPinned).map(([bucket, items]) => (
                        <div key={`pinned-${bucket}`}>
                          <div className="mb-3 text-xs uppercase tracking-wide text-neutral-500">
                            {bucket.replaceAll("_", " ")}
                          </div>

                          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                            {items.map((item) => (
                              <ProductCard
                                key={`pinned-${item.product_handle}`}
                                item={item}
                                pinned
                                onImageError={() => markBrokenImage(item.product_handle)}
                              />
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {visibleDisplayedProducts.length > 0 && (
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-sm font-semibold uppercase tracking-wide text-neutral-700">
                          More products
                        </h3>
                        <p className="mt-1 text-xs text-neutral-500">
                          Current rotated catalog page
                        </p>
                      </div>

                      {Object.entries(groupedShortlist).map(([bucket, items]) => (
                        <div key={bucket}>
                          <div className="mb-3 flex items-center justify-between">
                            <h3 className="text-sm font-semibold uppercase tracking-wide text-neutral-700">
                              {bucket.replaceAll("_", " ")}
                            </h3>
                            <span className="text-xs text-neutral-500">
                              {items.length} item(s)
                            </span>
                          </div>

                          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                            {items.map((item) => (
                              <ProductCard
                                key={`${item.bucket}-${item.product_handle}`}
                                item={item}
                                onImageError={() => markBrokenImage(item.product_handle)}
                              />
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

function ImageCard({
  title,
  image,
  emptyText,
}: {
  title: string;
  image: string | null;
  emptyText: string;
}) {
  return (
    <div className="overflow-hidden rounded-3xl border border-neutral-200 bg-white shadow-sm">
      <div className="border-b border-neutral-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-neutral-800">{title}</h3>
      </div>

      <div className="aspect-[4/3] bg-neutral-100">
        {image ? (
          <img src={image} alt={title} className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center px-6 text-center text-sm text-neutral-500">
            {emptyText}
          </div>
        )}
      </div>
    </div>
  );
}

function ProductCard({
  item,
  pinned = false,
  onImageError,
}: {
  item: RetrievedProduct;
  pinned?: boolean;
  onImageError?: () => void;
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-neutral-200 bg-white">
      <div className="aspect-[4/3] bg-neutral-100">
        {item.image_url ? (
          <img
            src={item.image_url}
            alt={item.title}
            className="h-full w-full object-cover"
            onError={() => onImageError?.()}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-neutral-500">
            No image
          </div>
        )}
      </div>

      <div className="p-4">
        <div className="flex flex-wrap items-center gap-2">
          {pinned && (
            <span className="rounded-full bg-amber-100 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-amber-800">
              In your room
            </span>
          )}
          {item.source === "innovative" && (
            <span className="rounded-full bg-sky-100 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-sky-800">
              Innovative product
            </span>
          )}
        </div>

        <div className="mt-2 line-clamp-2 text-sm font-medium text-neutral-900">
          {item.title}
        </div>

        <div className="mt-2 text-xs text-neutral-500">
          {item.normalized_category || item.category || "unknown"}
        </div>

        <div className="mt-2 text-xs text-neutral-500">
          Similarity: {item.similarity.toFixed(4)}
        </div>

        {(item.min_price !== null || item.max_price !== null) && (
          <div className="mt-2 text-sm font-semibold text-neutral-800">
            {item.min_price !== null && item.max_price !== null
              ? item.min_price === item.max_price
                ? `$${item.min_price}`
                : `$${item.min_price} - $${item.max_price}`
              : item.min_price !== null
                ? `$${item.min_price}`
                : `$${item.max_price}`}
          </div>
        )}
      </div>
    </div>
  );
}