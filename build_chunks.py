import json
import uuid 
import config

def create_guaranteed_unique_chunks(doctors_data, max_words_per_chunk, bio_word_limit):
    """
    Creates fully enriched chunks with a guaranteed unique ID (UUID) for each chunk.
    """
    processed_chunks = []
    
    for doctor in doctors_data:
        # --- Prepare Metadata ---
        metadata = doctor.copy()
        original_doctor_code = metadata.pop("code", None)
        bio_text_original = metadata.pop("bio", "") or ""
        comments_original = metadata.pop("comments_list", None)
        metadata['doctor_code'] = original_doctor_code
        
        bio_words = bio_text_original.split()
        truncated_bio = " ".join(bio_words[:bio_word_limit])
        
        speciality = doctor.get("speciality", "")
        base_text_prefix = f"تخصص: {speciality}\nبیو: {truncated_bio}"
        base_prefix_word_count = len(base_text_prefix.split())

        comments = comments_original if comments_original else []
        
        if not comments:
            processed_chunks.append({
                "chunk_id": str(uuid.uuid4()), 
                "chunk_text": base_text_prefix,
                "metadata": metadata
            })
            continue

        current_comments_list = []
        current_word_count = base_prefix_word_count
        
        for comment in comments:
            comment_word_count = len(comment.split())
            
            if (current_word_count + comment_word_count + 1) > max_words_per_chunk and current_comments_list:
                comments_text = "نظرات: " + ", ".join(current_comments_list)
                final_chunk_text = f"{base_text_prefix}\n{comments_text}"
                
                processed_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_text": final_chunk_text,
                    "metadata": metadata
                })
                
                current_comments_list = []
                current_word_count = base_prefix_word_count

            current_comments_list.append(comment)
            current_word_count += comment_word_count

        if current_comments_list:
            comments_text = "نظرات: " + ", ".join(current_comments_list)
            final_chunk_text = f"{base_text_prefix}\n{comments_text}"
            
            processed_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": final_chunk_text,
                "metadata": metadata
            })

    return processed_chunks

# --- Main execution block using variables from config ---
if __name__ == "__main__":
    try:
        # Read the raw data file path from config
        with open(config.RAW_DATA_PATH, 'r', encoding='utf-8') as f:
            all_doctors = json.load(f)
            
        print(f"Read {len(all_doctors)} doctors from '{config.RAW_DATA_PATH}'.")
        
        # Use the parameters from the config file
        final_chunks = create_guaranteed_unique_chunks(
            all_doctors, 
            max_words_per_chunk=config.MAX_WORDS_PER_CHUNK, 
            bio_word_limit=config.BIO_WORD_LIMIT
        )
        
        # Write to the chunked data file path from config
        with open(config.CHUNKED_DATA_PATH, 'w', encoding='utf-8') as f:
            for chunk in final_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
        print(f"Successfully created {len(final_chunks)} final chunks in '{config.CHUNKED_DATA_PATH}'.")

    except Exception as e:
        print(f"An error occurred: {e}")