import pdfplumber
import re
from pathlib import Path
import pandas as pd
import numpy as np

SECTION_KEYWORDS = {
    "WORK_EXPERIENCE": [
        "work experience", "experience", "employment", "work history", 
        "professional experience", "internship"
    ],
    "EDUCATION": [
        "education", "academic", "qualification", "educational background"
    ],
    "SKILLS": [
        "skills", "technical skills", "hard skills", "soft skills", 
        "competencies", "technologies", "expertise"
    ],
    "PROJECTS": [
        "projects", "personal projects", "academic projects", "key projects"
    ],
    "CAREER_OBJECTIVE": [
        "career objective", "objective", "summary", "profile", "about me", "professional summary"
    ]
}

def classify_block(block_text, block_idx=1, is_first_page=True):
    lines = [line.strip() for line in block_text.split('\n') if line.strip()]
    if not lines:
        return "OTHERS"
        
    first_line_clean = lines[0].lower()
    first_line_clean = re.sub(r'[^a-z\s]', '', first_line_clean).strip()

    if block_idx == 1 and ("@" in block_text or re.search(r'\d{3}[-\s]?\d{3}[-\s]?\d{4}', block_text)):
        return "HEADER"

    for category, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if first_line_clean == kw or first_line_clean.startswith(kw):
                return category

    if "@" in block_text or "linkedin" in block_text.lower() or "github" in block_text.lower():
        return "HEADER"
                    
    return "UNKNOWN"


def merge_blocks_by_type(classified_blocks):
    merged_results = {}
    current_category = "HEADER"

    for block in classified_blocks:
        category = block["category"]
        
        if category == "UNKNOWN":
            category = current_category
        else:
            current_category = category

        if category not in merged_results:
            merged_results[category] = {
                "category": category,
                "text": block["text"],
                "bounding_box": block["bounding_box"]
            }
        else:
            merged_results[category]["text"] += "\n\n" + block["text"]
            
            b1 = merged_results[category]["bounding_box"]
            b2 = block["bounding_box"]
            merged_results[category]["bounding_box"] = {
                "x0": min(b1["x0"], b2["x0"]),
                "top": min(b1["top"], b2["top"]),
                "x1": max(b1["x1"], b2["x1"]),
                "bottom": max(b1["bottom"], b2["bottom"])
            }

    return list(merged_results.values())
    
def find_split_x_by_density(words, page_width):
    #counter by pt
    x_counts = [0] * int(page_width)
    
    for w in words:
        x0 = max(0, int(w['x0']))
        x1 = min(int(page_width) - 1, int(w['x1']))
        for x in range(x0, x1 + 1):
            x_counts[x] += 1

    #searching range:0.25~0.75
    start_x = int(page_width * 0.25)
    end_x = int(page_width * 0.75)
    search_counts = x_counts[start_x:end_x]
    
    if not search_counts:
        return None, "single column"

    #the lowest words density
    min_density = min(search_counts)

    #average density form left and right (to prove the right split choose) 
    avg_left_density = sum(x_counts[int(page_width*0.05):start_x]) / max(1, (start_x - int(page_width*0.05)))
    avg_right_density = sum(x_counts[end_x:int(page_width*0.95)]) / max(1, (int(page_width*0.95) - end_x))
    avg_side_density = (avg_left_density + avg_right_density) / 2

    #avoid sparse words region
    if min_density > avg_side_density * 0.3:
        return None, "single column"

    #find width of gap
    threshold = min_density + 1
    best_gap_start = None
    best_gap_len = 0
    current_start = None

    for x in range(start_x, end_x):
        if x_counts[x] <= threshold:
            if current_start is None:
                current_start = x
        else:
            if current_start is not None:
                gap_len = x - current_start
                if gap_len > best_gap_len:
                    best_gap_len = gap_len
                    best_gap_start = current_start
                current_start = None

    if current_start is not None:
        gap_len = end_x - current_start
        if gap_len > best_gap_len:
            best_gap_len = gap_len
            best_gap_start = current_start

    #avoid normal words gap
    if best_gap_len < 8:
        return None, "single column"

    split_x = best_gap_start + (best_gap_len / 2)
    return split_x, "double column"

#separate blocks
def cluster_lines_to_blocks(words, block_gap_threshold=15):
    if not words:
        return []
        
    words = sorted(words, key=lambda w: (w['top'], w['x0']))
    lines, current_line, current_top = [], [], None
    
    for w in words:
        if current_top is None or abs(w['top'] - current_top) < 4:
            current_line.append(w)
            current_top = w['top'] if current_top is None else current_top
        else:
            lines.append(current_line)
            current_line, current_top = [w], w['top']
    if current_line:
        lines.append(current_line)

    blocks, current_block = [], []
    for line in lines:
        if not current_block:
            current_block.append(line)
        else:
            last_bottom = max(w['bottom'] for w in current_block[-1])
            line_top = min(w['top'] for w in line)
            
            if (line_top - last_bottom) < block_gap_threshold:
                current_block.append(line)
            else:
                blocks.append(current_block)
                current_block = [line]
    if current_block:
        blocks.append(current_block)
        
    return blocks


def parse_resume_by_x_density(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        
        if not words:
            print("No words")
            return

        split_x, layout_type = find_split_x_by_density(words, page.width)
        #print(f"column: {layout_type}" + (f" (split line x = {split_x:.1f})" if split_x else ""))

        raw_blocks_info = []

        if layout_type == "double column":
            header_words, left_words, right_words = [], [], []
            for w in words:
                if w['x0'] < (split_x - 10) and w['x1'] > (split_x + 10):
                    header_words.append(w)
                elif w['x1'] <= split_x + 5:
                    left_words.append(w)
                else:
                    right_words.append(w)

            all_sections = [header_words, left_words, right_words]
        else:
            all_sections = [words]

        raw_idx = 1
        for sec_words in all_sections:
            if not sec_words: continue
            blocks = cluster_lines_to_blocks(sec_words)
            for b in blocks:
                b_words = [w for line in b for w in line]
                text = "\n".join(" ".join(w['text'] for w in line) for line in b)
                bbox = {
                    "x0": round(min(w['x0'] for w in b_words), 1),
                    "top": round(min(w['top'] for w in b_words), 1),
                    "x1": round(max(w['x1'] for w in b_words), 1),
                    "bottom": round(max(w['bottom'] for w in b_words), 1)
                }
                
                category = classify_block(text, block_idx=raw_idx)
                raw_blocks_info.append({
                    "raw_id": raw_idx,
                    "category": category,
                    "text": text,
                    "bounding_box": bbox
                })
                raw_idx += 1

        final_sections = merge_blocks_by_type(raw_blocks_info)
        return final_sections

def extract_resume_text(file):
    sections = parse_resume_by_x_density(file)

    if not sections:
        return "" 

    resume = "\n".join(
        section["text"]
        for section in sections 
        if section.get("text")
    )
    return resume

def extract_resume_features(merged_sections):
    section_map = {sec['category']: sec['text'] for sec in merged_sections}
    
    skills_text = section_map.get("SKILLS", "")
    if skills_text:
        skills_text = skills_text.replace("\n", ", ")
        noise_pattern = r'\b(skills?|hard\s+skills?|soft\s+skills?|technical\s+skills?)\b'
        skills_text = re.sub(noise_pattern, '', skills_text, flags=re.IGNORECASE)
        skills_text = re.sub(r'\s*,\s*', ', ', skills_text)
        skills_text = re.sub(r'^\s*,\s*|\s*,\s*$', '', skills_text)
        skills_text = re.sub(r'(,\s*){2,}', ', ', skills_text)
        skills_text = skills_text.strip(" ,")
    
        if not skills_text:
            skills_text = np.nan
    else:
        skills_text = np.nan
    
    exp_text = section_map.get("WORK_EXPERIENCE", "") + "\n" + section_map.get("PROJECTS", "")
    years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', exp_text)]
    
    if len(years) >= 2:
        experience_years = max(years) - min(years)
    elif len(years) == 1:
        experience_years = max(0, 2026 - years[0])
    else:
        experience_years = 0

    edu_text = section_map.get("EDUCATION", "").lower()
    education_abbr = np.nan
    
    if re.search(r'\b(ph\.?d|doctor|doctoral)\b', edu_text):
        education_abbr = "PhD"
    elif re.search(r'\bmba\b', edu_text):
        education_abbr = "MBA"
    elif re.search(r'\b(m\.?tech|master\s+of\s+technology)\b', edu_text):
        education_abbr = "M.Tech"
    elif re.search(r'\b(m\.?sc|m\.?s\b|m\.?a\b|master)\b', edu_text):
        education_abbr = "M.Sc"
    elif re.search(r'\b(b\.?tech|bachelor\s+of\s+technology)\b', edu_text):
        education_abbr = "B.Tech"
    elif re.search(r'\b(b\.?sc|b\.?s\b|b\.?a\b|b\.?e\b|bachelor)\b', edu_text):
        education_abbr = "B.Sc"
    else:
        education_abbr = np.nan

    cert_source_text = (section_map.get("OTHERS", "") + "\n" + section_map.get("CAREER_OBJECTIVE", ""))
    cert_keywords = [
        "AWS Certified", "Google ML", "Deep Learning Specialization", 
        "PMP", "CKA", "CISSP", "Azure Certified"
    ]
    found_certs = []
    
    for kw in cert_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', cert_source_text, re.IGNORECASE):
            found_certs.append(kw)
            
    if not found_certs and cert_source_text:
        cert_lines = [
            line.strip() for line in cert_source_text.split('\n') 
            if any(term in line.lower() for term in ["certif", "license", "credential"])
        ]
        
        if cert_lines:
            found_certs.extend(cert_lines)

    certifications = ", ".join(found_certs) if found_certs else np.nan
    
    def count_items_by_blank_line(text):
        if not text.strip():
            return 0
        items = [item for item in re.split(r'\n\s*\n', text.strip()) if item.strip()]
        return len(items)

    work_count = count_items_by_blank_line(section_map.get("WORK_EXPERIENCE", ""))
    proj_count = count_items_by_blank_line(section_map.get("PROJECTS", ""))
    total_projects_count = work_count + proj_count
    
    return {
        "Skills": skills_text,
        "Experience (Years)": experience_years,
        "Education": education_abbr,
        "Certifications": certifications,
        "Projects Count": total_projects_count
    }

def create_resume_dataframe(all_parsed_resumes):
    rows = []
    for sections in all_parsed_resumes:
        row = extract_resume_features(sections)
        rows.append(row)
        
    columns_order = [
        "Skills", 
        "Experience (Years)", 
        "Education", 
        "Certifications", 
        "Projects Count"
    ]
    
    df = pd.DataFrame(rows)[columns_order]
    return df

if __name__ == "__main__":
    input_directory = Path("raw_resume")
    files = [
        f for f in input_directory.iterdir() 
        if f.is_file()
    ]
    all_parsed_resumes = []
    for path in files:
        final_sections = parse_resume_by_x_density(path)
        all_parsed_resumes.append(final_sections)
        
    df = create_resume_dataframe(all_parsed_resumes)
    print(df)
    df.to_csv("data.csv", index=False)