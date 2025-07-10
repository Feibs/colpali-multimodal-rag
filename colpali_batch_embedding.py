#!/usr/bin/env python3
"""
ColPali Batch Embedding Script

This script processes PDF files using ColPali embeddings, converts them to images,
generates multi-vector embeddings via the ColPali embedding service,
and stores them in a Qdrant vector database with multi-vector configuration.
"""

import os
import requests
import io
import base64
import argparse
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.http import models
import stamina

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "colnomic_documents"

# ColPali Embedding Service configuration
COLPALI_API_ENDPOINT = os.getenv("COLPALI_API_ENDPOINT", "http://localhost:8001")

def setup_qdrant():
    """
    Set up the Qdrant collection for ColPali multi-vector embeddings.
    """
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        print(f"Creating collection {COLLECTION_NAME} with multi-vector configuration")
        
        # Create collection with multi-vector configuration (similar to ColPali demo)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            on_disk_payload=True,  # Store payload on disk for efficiency
            vectors_config=models.VectorParams(
                size=128,  # ColPali vector dimension
                distance=models.Distance.COSINE,
                on_disk=True,  # Move original vectors to disk
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True  # Keep only quantized vectors in RAM
                    ),
                ),
            ),
        )
        
        # Create payload indexes for efficient filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="title",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="page_number",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        
        print(f"Collection {COLLECTION_NAME} created successfully with multi-vector support")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
    
    return client

def get_colpali_embeddings_from_api(images: List[Image.Image], batch_size: int = 4) -> List[List[List[float]]]:
    """
    Get ColPali multi-vector embeddings from the API for a batch of images.
    
    Args:
        images: List of PIL Image objects
        batch_size: Number of images to process in each API call
        
    Returns:
        List of multi-vector embeddings (each image gets multiple vectors)
    """
    all_embeddings = []
    
    print(f"Processing {len(images)} images in batches of {batch_size}")
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Prepare files for API request
        files = []
        for j, img in enumerate(batch_images):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            files.append(('files', (f'image{j}.png', img_byte_arr, 'image/png')))
        
        # Make API request to ColPali service
        try:
            print(f"Sending batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1} to ColPali API...")
            response = requests.post(f"{COLPALI_API_ENDPOINT}/embed_images", files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result['embeddings']
                all_embeddings.extend(embeddings)
                print(f"✓ Received {len(embeddings)} multi-vector embeddings")
            else:
                print(f"✗ Error getting embeddings: {response.status_code} - {response.text}")
                # Add empty embeddings as placeholders
                all_embeddings.extend([[] for _ in batch_images])
        except Exception as e:
            print(f"✗ Exception calling ColPali API: {str(e)}")
            # Add empty embeddings as placeholders
            all_embeddings.extend([[] for _ in batch_images])
    
    return all_embeddings

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(client: QdrantClient, points: List[models.PointStruct]):
    """
    Upsert points to Qdrant with retry logic.
    """
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=False,
        )
        return True
    except Exception as e:
        print(f"Error during upsert: {e}")
        raise

def get_version(title: str) -> str:
    index = title.rfind('v')
    version = 'v00'
    if index != -1:
        version = title[index:]
    return version

def process_pdf(pdf_info: Dict[str, str], client: QdrantClient) -> None:
    """
    Process a PDF file using ColPali, convert to images, generate multi-vector embeddings, 
    and store in Qdrant.
    
    Args:
        pdf_info: Dictionary containing 'title' and 'file' keys
        client: Qdrant client instance
    """
    title = pdf_info['title']
    file_path = pdf_info['file']
    
    print(f"\n🔄 Processing PDF: {title}")
    print(f"   Source: {file_path}")
    
    # Download PDF if it's a URL
    if file_path.startswith('http'):
        local_path = f"{title.replace(' ', '_')}.pdf"
        print(f"📥 Downloading PDF from {file_path}")
        try:
            with open(local_path, 'wb') as f:
                f.write(requests.get(file_path, timeout=120).content)
            file_path = local_path
            print(f"✓ Downloaded to {local_path}")
        except Exception as e:
            print(f"✗ Error downloading PDF: {str(e)}")
            return
    
    # Convert PDF to images
    print(f"🖼️  Converting PDF to images...")
    try:
        images = convert_from_path(file_path, fmt="png", dpi=200)
        print(f"✓ Converted {len(images)} pages to images")
    except Exception as e:
        print(f"✗ Error converting PDF to images: {str(e)}")
        return
    
    # Generate ColPali embeddings
    print(f"🧮 Generating ColPali multi-vector embeddings...")
    embeddings = get_colpali_embeddings_from_api(images)
    
    if not embeddings or all(not emb for emb in embeddings):
        print(f"✗ Failed to generate embeddings for {title}")
        return
    
    print(f"✓ Generated {len(embeddings)} multi-vector embeddings")
    
    # Store in Qdrant
    print(f"💾 Storing embeddings in Qdrant...")
    points = []
    
    for page_idx, (image, embedding) in enumerate(zip(images, embeddings)):
        if not embedding:  # Skip empty embeddings
            print(f"⚠️  Skipping page {page_idx + 1} due to empty embedding")
            continue
            
        page_number = page_idx + 1
        
        # Convert image to base64 for storage
        image_base64 = encode_image_to_base64(image)
        
        # Create point with multi-vector embedding and payload
        # Use a hash-based ID to avoid collisions
        point_id = int(f"{hash(title) % 1000000000}{page_number:04d}")
        
        points.append(models.PointStruct(
            id=point_id,
            vector=embedding,  # This is now a multi-vector (list of vectors)
            payload={
                "title": title,
                "file_path": file_path,
                "doc_version": get_version(title),
                "page_number": page_number,
                "image_data": image_base64,
                "created_at": int(time.time()),
                "total_pages": len(images),
                "embedding_type": "colpali_multivector"
            }
        ))
    
    # Upsert points in batches to avoid timeout
    batch_size = 5
    total_batches = (len(points) - 1) // batch_size + 1
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        try:
            upsert_to_qdrant(client, batch_points)
            print(f"✓ Stored batch {i//batch_size + 1}/{total_batches}")
        except Exception as e:
            print(f"✗ Error storing batch {i//batch_size + 1}: {str(e)}")
    
    print(f"✅ Successfully processed {title}: {len(points)} pages stored")

def main():
    """
    Main function to process PDFs from command line arguments or config file.
    """
    # Declare global variables at the start of the function
    global QDRANT_URL, COLPALI_API_ENDPOINT
    
    parser = argparse.ArgumentParser(description='Process PDFs with ColPali and store embeddings in Qdrant')
    parser.add_argument('--config', type=str, help='Path to config JSON file with PDFs to process')
    parser.add_argument('--pdf', type=str, help='Path to a single PDF file')
    parser.add_argument('--title', type=str, help='Title for the PDF (required if --pdf is used)')
    parser.add_argument('--qdrant-url', type=str, default=QDRANT_URL, help='Qdrant server URL')
    parser.add_argument('--colpali-api', type=str, default=COLPALI_API_ENDPOINT, help='ColPali API endpoint')
    args = parser.parse_args()
    
    # Update global variables if provided
    QDRANT_URL = args.qdrant_url
    COLPALI_API_ENDPOINT = args.colpali_api
    
    print("🚀 ColPali Batch Embedding Script")
    print(f"   Qdrant URL: {QDRANT_URL}")
    print(f"   ColPali API: {COLPALI_API_ENDPOINT}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    # Test ColPali API connection
    try:
        response = requests.get(f"{COLPALI_API_ENDPOINT}/health", timeout=10)
        if response.status_code == 200:
            health_info = response.json()
            print(f"✓ ColPali API is healthy: {health_info}")
        else:
            print(f"⚠️  ColPali API returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to ColPali API: {str(e)}")
        print("Please ensure the ColPali embedding service is running")
        return
    
    # Setup Qdrant collection
    try:
        client = setup_qdrant()
        print("✓ Qdrant connection established")
    except Exception as e:
        print(f"✗ Error setting up Qdrant: {str(e)}")
        return
    
    # Process PDFs
    if args.config:
        # Process PDFs from config file
        print(f"\n📄 Loading PDFs from config: {args.config}")
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            pdfs = config.get('PDFs', [])
            print(f"Found {len(pdfs)} PDFs to process")
            
            for i, pdf_info in enumerate(pdfs, 1):
                print(f"\n--- Processing PDF {i}/{len(pdfs)} ---")
                process_pdf(pdf_info, client)
                
        except Exception as e:
            print(f"✗ Error reading config file: {str(e)}")
            return
            
    elif args.pdf and args.title:
        # Process a single PDF
        print(f"\n📄 Processing single PDF")
        pdf_info = {
            'title': args.title,
            'file': args.pdf
        }
        process_pdf(pdf_info, client)
        
    else:
        # Default PDFs for demonstration
        print(f"\n📄 Processing default PDFs")
        pdfs = [
            {'title': "Chinese Cook Book", 'file': "https://www.retigo.com/userfiles/dokumenty_menu/193/retigo-chinese_cookbook_final.pdf"}
        ]
        
        for pdf_info in pdfs:
            process_pdf(pdf_info, client)
    
    # Update collection optimizer settings
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
        )
        print("✓ Updated collection optimizer settings")
    except Exception as e:
        print(f"⚠️  Could not update optimizer settings: {str(e)}")
    
    print("\n🎉 Batch processing complete!")

if __name__ == "__main__":
    main() 