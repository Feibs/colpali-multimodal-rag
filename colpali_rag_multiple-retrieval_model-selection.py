#!/usr/bin/env python3
"""
ColPali RAG Multiple Retrieval Pipeline

This pipeline uses ColPali multi-vector embeddings for document retrieval,
retrieves TOP_K relevant documents and sends all their images to VLM for
comprehensive answer generation. Compatible with OpenWebUI pipeline format.
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests
import numpy as np
from PIL import Image
import io
import base64
import ast
from qdrant_client import QdrantClient
from qdrant_client.http import models

class Pipeline:
    class Valves(BaseModel):
        COLPALI_API_ENDPOINT: str
        VLM_API_ENDPOINT: str
        VLM_API_KEY: str
        VLM_MODEL_ID: str
        CUSTOM_LLM_API_ENDPOINT: str
        CUSTOM_LLM_API_KEY: str
        CUSTOM_LLM_MODEL_ID: str
        QDRANT_URL: str
        COLLECTION_NAME: str
        VLM_SYS_PROMPT: str
        CUSTOM_LLM_SYS_PROMPT: str
        TOP_K: int
        GAP_BASED_THRESHOLD: float
        ADAPTIVE_THRESHOLD: float

    def __init__(self):
        self.name = "People and Policy Hub"
        self.valves = self.Valves(
            COLPALI_API_ENDPOINT=os.getenv("COLPALI_API_ENDPOINT", "http://my-nomic-embedding-service"),
            VLM_API_ENDPOINT=os.getenv("VLM_API_ENDPOINT", "http://10.16.0.4:4000/v1"),
            VLM_API_KEY=os.getenv("VLM_API_KEY", ""),
            VLM_MODEL_ID=os.getenv("VLM_MODEL_ID", "gpt-4o"),
            CUSTOM_LLM_API_ENDPOINT=os.getenv("CUSTOM_LLM_API_ENDPOINT", "http://10.16.0.4:3001/api"),
            CUSTOM_LLM_API_KEY=os.getenv("CUSTOM_LLM_API_KEY", ""),
            CUSTOM_LLM_MODEL_ID=os.getenv("CUSTOM_LLM_MODEL_ID", "hr-model"),
            QDRANT_URL=os.getenv("QDRANT_URL", "http://my-qdrant-url"),
            COLLECTION_NAME=os.getenv("QDRANT_COLLECTION", "my_collection"),
            VLM_SYS_PROMPT=os.getenv("VLM_SYS_PROMPT", "Anda adalah seorang analis dokumen ahli dengan pengalaman luas dalam analisis lintas dokumen dan sintesis informasi. Tugas Anda adalah:  1. FASE ANALISIS: - Analisis setiap gambar dokumen yang diberikan secara individual - Identifikasi informasi kunci, termasuk tanggal, angka, topik utama, dan detail penting - Catat setiap hubungan atau kontradiksi antar dokumen  2. FASE RINGKASAN: - Berikan ringkasan singkat untuk setiap dokumen - Buat ringkasan terpadu yang menyoroti tema umum dan kata kunci - Tunjukkan kualitas/kejelasan gambar dan setiap keterbatasan dalam membacanya  3. FASE JAWABAN PERTANYAAN: - Jawab pertanyaan spesifik dari pengguna berdasarkan bukti dari dokumen, perhatikan kata kunci antara pertanyaan pengguna dan dokumen - Kutip referensi spesifik menggunakan pengidentifikasi dokumen (misalnya, 'Dokumen A menyatakan...') - Soroti di mana beberapa dokumen menjawab pertanyaan pengguna - Tunjukkan dengan jelas jika ada informasi yang diperlukan yang hilang atau tidak jelas  Format jawaban Anda dengan: - Judul bagian yang jelas - Poin-poin untuk informasi kunci - Kutipan langsung ketika sangat relevan - Referensi silang antar dokumen  Jika Anda menemui keterbatasan dalam kualitas gambar atau kejelasan konten, harap nyatakan keterbatasan tersebut secara eksplisit dalam analisis Anda."),
            CUSTOM_LLM_SYS_PROMPT=os.getenv("CUSTOM_LLM_SYS_PROMPT", "You are a specialized assistant. If the question is out of your knowledge base, only reply exactly with `-`. You must only answer based on your knowledge base."),
            TOP_K=int(os.getenv("TOP_K", "3")),
            GAP_BASED_THRESHOLD=os.getenv("GAP_BASED_THRESHOLD", "0.1"),
            ADAPTIVE_THRESHOLD=os.getenv("ADAPTIVE_THRESHOLD", "0.6")
        )
        
        self.client = None
        self.initialized = False

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        # We'll initialize the connection in the first call to avoid slowing down startup

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def get_qdrant_client(self):
        """Get a connection to the Qdrant vector database"""
        if not self.client:
            self.client = QdrantClient(url=self.valves.QDRANT_URL, port=None)
        return self.client

    def initialize_data(self):
        """Initialize the connection to Qdrant and verify ColPali API"""
        if self.initialized:
            return
            
        print("Initializing ColPali RAG Multiple Retrieval Pipeline...")
        
        # Test ColPali API connection
        try:
            response = requests.get(f"{self.valves.COLPALI_API_ENDPOINT}/health", timeout=10)
            if response.status_code == 200:
                health_info = response.json()
                print(f"✓ ColPali API is healthy: {health_info}")
            else:
                print(f"⚠️  ColPali API returned status {response.status_code}")
                raise ValueError(f"ColPali API not healthy: {response.status_code}")
        except Exception as e:
            print(f"✗ Cannot connect to ColPali API: {str(e)}")
            raise ValueError(f"ColPali API connection failed: {str(e)}")
        
        # Test Qdrant connection
        try:
            client = self.get_qdrant_client()
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.valves.COLLECTION_NAME in collection_names:
                # Get collection info
                collection_info = client.get_collection(collection_name=self.valves.COLLECTION_NAME)
                points_count = client.count(collection_name=self.valves.COLLECTION_NAME).count
                
                # Check if it's a multi-vector collection
                is_multivector = hasattr(collection_info.config.params.vectors, 'multivector_config')
                multivector_info = ""
                if is_multivector:
                    mv_config = collection_info.config.params.vectors.multivector_config
                    multivector_info = f" (Multi-vector: {mv_config.comparator})"
                
                print(f"✓ Connected to Qdrant. Collection '{self.valves.COLLECTION_NAME}' has {points_count} points{multivector_info}.")
                print(f"✓ Configured to retrieve TOP_K={self.valves.TOP_K} documents per query")
                self.initialized = True
            else:
                print(f"✗ Collection '{self.valves.COLLECTION_NAME}' not found. Please run the batch embedding script first.")
                raise ValueError(f"Collection '{self.valves.COLLECTION_NAME}' not found in Qdrant")
                
        except Exception as e:
            print(f"✗ Error connecting to Qdrant: {str(e)}")
            raise
            
        print("✅ ColPali RAG Multiple Retrieval Pipeline initialization complete")

    def gap_based_threshold(self, results: list, gap_threshold: float = 0.1) -> list:
        """
        Apply gap-based threshold filtering to results.
        
        Gap-based threshold identifies a significant drop in similarity scores
        and filters out results after the gap, assuming they are less relevant.
        
        Args:
            results: List of search results with similarity scores
            gap_threshold: Minimum gap between consecutive scores to trigger cutoff
            
        Returns:
            Filtered list of results
        """
        if len(results) <= 1:
            return results
        
        filtered_results = [results[0]]  # Always keep the top result
        
        for i in range(1, len(results)):
            current_score = results[i]["similarity"]
            previous_score = results[i-1]["similarity"]
            
            # Calculate the gap between consecutive scores
            gap = previous_score - current_score
            
            # If gap is significant, stop including more results
            if gap > gap_threshold:
                print(f"🔍 Gap-based threshold: Found significant gap of {gap:.4f} after rank {i}")
                break
            
            filtered_results.append(results[i])
        
        return filtered_results
    
    def adaptive_threshold(self, results: list, std_multiplier: float = 1.0) -> list:
        """
        Apply adaptive threshold filtering based on statistical analysis of scores.
        
        Adaptive threshold calculates the mean and standard deviation of similarity scores
        and filters out results that fall below (mean - std_multiplier * std_dev).
        
        Args:
            results: List of search results with similarity scores
            std_multiplier: Multiplier for standard deviation in threshold calculation
            
        Returns:
            Filtered list of results
        """
        if len(results) <= 1:
            return results
        
        # Extract similarity scores
        scores = [result["similarity"] for result in results]
        
        # Calculate statistical measures
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Calculate adaptive threshold
        adaptive_threshold_value = mean_score - (std_multiplier * std_score)
        
        print(f"🔍 Adaptive threshold: mean={mean_score:.4f}, std={std_score:.4f}, threshold={adaptive_threshold_value:.4f}")
        
        # Filter results based on adaptive threshold
        filtered_results = []
        for result in results:
            if result["similarity"] >= adaptive_threshold_value:
                filtered_results.append(result)
            else:
                print(f"🔍 Adaptive threshold: Filtered out result with score {result['similarity']:.4f}")
                break  # Stop at first result below threshold (assuming sorted order)
        
        return filtered_results

    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve semantically similar items from Qdrant using ColPali multi-vector embeddings"""
        
        # Get query embedding from ColPali API
        try:
            response = requests.post(
                f"{self.valves.COLPALI_API_ENDPOINT}/embed_query", 
                json={"query": query},
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"✗ Error getting query embedding: {response.status_code} - {response.text}")
                return []
            
            query_data = response.json()
            query_embedding = query_data['embedding']  # This is a multi-vector embedding
            
            print(f"🔍 Generated query embedding with {len(query_embedding)} vectors")
            
        except Exception as e:
            print(f"✗ Exception getting query embedding: {str(e)}")
            return []
        
        # Query Qdrant for similar documents using multi-vector search
        client = self.get_qdrant_client()
        results = []
        
        try:
            print(f"🔎 Searching Qdrant for top {k} similar documents...")
            
            # Use query_points for multi-vector search (similar to ColPali demo)
            search_results = client.query_points(
                collection_name=self.valves.COLLECTION_NAME,
                query=query_embedding,  # Multi-vector query
                limit=k,
                timeout=100,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                with_payload=True,
                with_vectors=False
            )
            
            print(f"✓ Found {len(search_results.points)} results")
            
            # Process results
            for i, result in enumerate(search_results.points):
                try:
                    # Convert base64 image data back to PIL Image
                    image_data = base64.b64decode(result.payload["image_data"])
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Create result object
                    results.append({
                        "id": str(result.id),
                        "title": result.payload["title"],
                        "file_path": result.payload["file_path"],
                        "page_number": result.payload["page_number"],
                        "image": image,
                        "similarity": result.score,
                        "total_pages": result.payload.get("total_pages", 1),
                        "embedding_type": result.payload.get("embedding_type", "unknown"),
                        "rank": i + 1
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error processing result {result.id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"✗ Error retrieving documents from Qdrant: {str(e)}")
        
        return results

    def encode_image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        # Resize if needed for display efficiency
        if image.width > 1024 or image.height > 1024:
            image = image.resize((1024, 768), Image.LANCZOS)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def encode_image_webp_to_base64(self, image):
        """Convert PIL image to WebP base64 string for efficient display"""
        # Resize proportionally if needed
        if image.width > 1024 or image.height > 1024:
            aspect_ratio = image.width / image.height
            new_height = 768
            new_width = int(new_height * aspect_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        buffered = io.BytesIO()
        image.save(buffered, format="WebP", quality=75)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def query_vlm_api(self, query: str, images: list, pages: list) -> str:
        """Queries VLM API with text and multiple images using OpenAI-compatible endpoint"""
        system_prompt = self.valves.VLM_SYS_PROMPT
        
        # Prepare the message content with images and query
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add all images to the content
        for i, image in enumerate(images):
            base64_image = self.encode_image_to_base64(image)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        
        # Add the text query with context about multiple documents
        query_text = f"Question: {query} \n\nYou are given a list of pages from a PDF document:{pages} \n\nEach page includes metadata such as its page number and an image of the page. The list is initially ordered by a similarity score, but I want you to independently evaluate the content of the pages (e.g., based on their text, layout, or visual cues) and re-rank them based on their relevance or importance. If a page is irrelevant or unhelpful, feel free to exclude it from the result. \n\nReturn your output as a plain text in this format: {{<page_number>: <final rank>}} \n\nOnly return the plain text 'dictionary'. Do not include any explanation or extra text. If you think all documents are not relevant to the query, return empty {{}}."

        messages[1]["content"].append({
            "type": "text",
            "text": query_text
        })
        
        # Prepare the API request
        payload = {
            "model": self.valves.VLM_MODEL_ID,
            "messages": messages,
            "max_tokens": 1500,  # Increased for multiple documents
            "temperature": 0.01,
            "top_p": 0.001,
        }

        headers = {
            "Authorization": f"Bearer {self.valves.VLM_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Make the API request
        try:
            print(f"🤖 Sending {len(images)} images to VLM for analysis...")
            response = requests.post(
                f"{self.valves.VLM_API_ENDPOINT}/chat/completions", 
                headers=headers,
                json=payload,
                timeout=180  # Increased timeout for multiple images
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"✗ Error from VLM API: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from VLM API. {response.text}. Status code: {response.status_code}"
        except Exception as e:
            print(f"✗ Exception when calling VLM API: {str(e)}")
            return f"Error: {str(e)}"

    def query_custom_llm_api(self, query: str) -> str:
        """Queries custom LLM API with default knowledge, e.g. video knowledge using OpenAI-compatible endpoint"""
        system_prompt = self.valves.CUSTOM_LLM_SYS_PROMPT
        
        # Prepare the message content with images and query
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Prepare the API request
        payload = {
            "model": self.valves.CUSTOM_LLM_MODEL_ID,
            "messages": messages,
            "max_tokens": 1500,  # Increased for multiple documents
            "temperature": 0.01,
            "top_p": 0.001,
        }

        headers = {
            "Authorization": f"Bearer {self.valves.CUSTOM_LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Make the API request
        try:
            response = requests.post(
                f"{self.valves.CUSTOM_LLM_API_ENDPOINT}/chat/completions", 
                headers=headers,
                json=payload,
                timeout=180  # Increased timeout for multiple images
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"✗ Error from custom LLM API: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from custom LLM API. {response.text}. Status code: {response.status_code}"
        except Exception as e:
            print(f"✗ Exception when calling custom LLM API: {str(e)}")
            return f"Error: {str(e)}"


    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        
        # Initialize data if not already done
        try:
            self.initialize_data()
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}\n\nPlease ensure:\n1. ColPali embedding service is running\n2. Qdrant is accessible\n3. Collection exists (run batch embedding script first)"
        
        try:
            # Process the query
            query = user_message
            print(f"🔍 Processing query: '{query}'")

            # By default answer with knowledge from the custom LLM
            response = self.query_custom_llm_api(query)
            if response != '-':           
                return response
            
            print(f"📊 Retrieving TOP_K={self.valves.TOP_K} relevant documents")
            
            # Retrieve relevant documents using ColPali
            results = self.retrieve(query, self.valves.TOP_K)
            
            if not results:
                return f"Sorry, no relevant documents found for your query. Please ensure documents have been indexed using the ColPali batch embedding script."
            
            print(f"📄 Found {len(results)} relevant document(s) before threshold filtering:")
            for result in results:
                print(f"  {result['rank']}. {result['title']}, Page {result['page_number']} (Score: {result['similarity']:.4f})")

            images = [result["image"] for result in results]

            try:
                answer = self.query_vlm_api(query, images, results)
                reranked_docs = ast.literal_eval(str(answer))

                if (len(reranked_docs) == 0):
                    return "Sorry, no relevant documents found for your question. Try adding more documents to your collection or ask another question."

                new_results = [0] * len(reranked_docs)
                doc_info = f"\n\nSee {len(reranked_docs)} page(s) below for this matter.\n"
                print(f"🤖 Generating re-ranked answer using VLM with {len(images)} images...")

                for result in results:
                    if isinstance(reranked_docs, dict) and result['page_number'] in reranked_docs:
                        new_rank = int(reranked_docs.get(result['page_number']))
                        result['rank'] = new_rank
                        new_results[new_rank-1] = result
                results = new_results

            except Exception as e:
                error_message = f"Sorry, let's try another question."
                answer = self.query_vlm_api(query, images, results)
                print(f"❌ Error query vlm api: {str(e)} \n\nanswer: {repr(answer)} | is dict? {str(isinstance(answer, dict))} | is str? {str(isinstance(answer, str))}")
                return error_message
            
            # Create the text response
            text_response = f"{doc_info}"
            
            # Return as an iterator that can be streamed
            def generate_response():
                yield text_response
                yield "\n\n"
                
                # Add all document images for reference
                yield f"📋 **Retrieved document page:**\n\n"
                for result in results:
                    b64_img = self.encode_image_webp_to_base64(result["image"])
                    yield f"**{result['rank']}. {result['title']} - Page {result['page_number']}**\n"
                    yield f"![Document Page {result['rank']}](data:image/webp;base64,{b64_img})\n\n"

            return generate_response()
                
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            error_message = "Sorry, an error occurred. Try again in a moment."
            return error_message

# For standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        pipeline = Pipeline()
        
        # Test query
        test_query = "What ingredients are needed for cooking?"
        
        print(f"Testing ColPali RAG Multiple Retrieval Pipeline with query: '{test_query}'")
        print(f"TOP_K setting: {pipeline.valves.TOP_K}")
        
        try:
            result = pipeline.pipe(test_query, "test", [], {})
            
            if isinstance(result, str):
                print("Result:", result)
            else:
                # Handle generator
                for chunk in result:
                    print(chunk, end="")
                print()
                
        except Exception as e:
            print(f"Test failed: {str(e)}")
    
    # Run test
    asyncio.run(test_pipeline()) 