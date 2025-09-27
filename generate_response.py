import json
import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from openai import OpenAI 
import config

# This map is fine here as it's part of the function's logic
CITY_MAP = {
    "تهران": "تهران", "طهران": "تهران", "کرج": "کرج", "تبریز": "تبریز", 
    "مشهد": "مشهد", "مشد": "مشهد", "شیراز": "شیراز", "شیرز": "شیراز",
    "اصفهان": "اصفهان", "اصفهون": "اصفهان", "اصفحان": "اصفهان"
}

def extract_city(query: str):
    for variation, standard_name in CITY_MAP.items():
        if variation in query: return standard_name
    return None

def search_and_rerank(collection, model, query_text: str):
    city_filter = extract_city(query_text)
    query_embedding = model.encode(query_text).tolist()
    
    where_clause = {}
    if city_filter:
        where_clause = {"city": city_filter}
        print(f"Filtering results for city: {city_filter}")

    # Use INITIAL_SEARCH_RESULTS from config
    n_results = config.INITIAL_SEARCH_RESULTS 
    if where_clause:
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_clause)
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
    if not results['ids'][0]:
        return []

    doctor_scores = defaultdict(lambda: {"chunk_count": 0, "total_distance": 0, "metadata": {}})
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        doctor_code = metadata.get('doctor_code')

        if not doctor_code: continue
        doctor_scores[doctor_code]['chunk_count'] += 1
        doctor_scores[doctor_code]['total_distance'] += distance
        if not doctor_scores[doctor_code]['metadata']:
            doctor_scores[doctor_code]['metadata'] = metadata

    final_ranked_list = []
    for code, data in doctor_scores.items():
        avg_distance = data['total_distance'] / data['chunk_count']
        rating_score = float(data['metadata'].get('rating_score') or 0)
        final_score = (data['chunk_count'] * 100) - avg_distance + (rating_score / 10)
        
        data['metadata']['final_score'] = final_score
        data['metadata']['chunk_count'] = data['chunk_count']
        final_ranked_list.append(data['metadata'])
        
    return sorted(final_ranked_list, key=lambda x: x['final_score'], reverse=True)

def query_llm_api(prompt: str):
    try:
        # Use API settings from config
        client = OpenAI(base_url=config.GAPGPT_API_BASE_URL, api_key=config.GAPGPT_API_KEY)
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API request failed: {e}")

def main():
    print("Loading models and connecting to DB...")
    # Use model name and DB path from config
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    db_client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = db_client.get_collection(name=config.COLLECTION_NAME)
    print("System is ready.")

    user_query = input("\nسلام! مشکل شما چیست و در کدام شهر به دنبال پزشک هستید؟\n> ")
    
    ranked_doctors = search_and_rerank(collection, embedding_model, user_query)
    
    if not ranked_doctors:
        print("متاسفانه پزشکی با مشخصات مورد نظر شما یافت نشد.")
        return

    # Use TOP_N_FOR_CONTEXT from config to decide how many doctors to use for context
    top_doctors_for_context = ranked_doctors[:config.TOP_N_FOR_CONTEXT]
    print(f"\nFound top doctors: {[doc.get('name') for doc in top_doctors_for_context]}")
    
    context_parts = []
    for i, doctor in enumerate(top_doctors_for_context):
        doctor_info = f"""
پزشک شماره {i+1}:
- نام: {doctor.get('name', 'N/A')}
- تخصص: {doctor.get('speciality', 'N/A')}
- آدرس: {doctor.get('address', 'N/A')}
"""
        context_parts.append(doctor_info)
        
    context = "\n---\n".join(context_parts)

    prompt_template = f"""
[دستورالعمل]
شما یک دستیار هوشمند برای پیشنهاد پزشک هستید. وظیفه شما عمل کردن به عنوان یک فیلتر نهایی و منطقی است.
1. سوال کاربر و لیست {config.TOP_N_FOR_CONTEXT} پزشک پیشنهادی زیر را با دقت بررسی کن.
2. **تخصص** هر پزشک را به طور مستقیم با نیاز بیان شده در سوال کاربر مقایسه کن.
3. حداکثر **{config.FINAL_RECOMMENDATIONS} پزشک** که تخصصشان بیشترین ارتباط را با سوال کاربر دارد، انتخاب کن.
4. برای هر پزشک منتخب، نام، تخصص و آدرس او را به صورت واضح و مرتب نمایش بده و در یک جمله کوتاه توضیح بده چرا انتخاب خوبی است.

[اطلاعات {config.TOP_N_FOR_CONTEXT} پزشک برتر]
{context}

[سوال کاربر]
{user_query}

[پاسخ نهایی شما به زبان فارسی]:
"""
    
    print(f"\nGenerating final recommendation with {config.LLM_MODEL}...")
    final_answer = query_llm_api(prompt_template)
    
    print("\n--- پیشنهاد هوشمند برای شما ---")
    print(final_answer.strip())

if __name__ == "__main__":
    main()