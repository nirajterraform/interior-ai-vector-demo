export type RoomType =
  | "living_room"
  | "bedroom"
  | "dining_room"
  | "kitchen"
  | "office";

export type ProductRow = {
  product_handle: string;
  title: string;
  category: string | null;
  subcategory: string | null;
  normalized_category: string | null;
  primary_image_url: string | null;
  min_price: number | null;
  max_price: number | null;
  similarity: number;
};

export type RetrievedProduct = {
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
};

export type RetrievalBucket = {
  bucket: string;
  categories: string[];
  limit: number;
  required: boolean;
};

export type RetrievalResponse = {
  roomType: RoomType;
  theme: string;
  shortlist: RetrievedProduct[];
};