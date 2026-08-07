import streamlit as st
from resume_reader_new import extract_resume_text
from data_cleaning import clean_text
from job_classifier import load_classifier, predict_job_role
from resume_matcher import calculate_match, compare_skills, format_skill_name

model, tfidf = load_classifier()

st.set_page_config(
    page_title = "🤖 ResuMatch",
    page_icon="📄",
    layout = "wide",
)

st.title("ResuMatch")
st.write("Upload a resume and paste a job description to analyze how much they're compatible")

upload_tab, results_tab = st.tabs(["Upload", "Results"])

with upload_tab: 
    st.subheader("Resume and Job Information")

    left_column, right_column = st.columns(2) 

    with left_column:
        resume_file = st.file_uploader(
            "Upload your resume",
            type = ["pdf", "docx", "txt"],
        )

    with right_column:
        job_description = st.text_area(
            "Paste the job description",
            height = 250,
            placeholder= "Paste the complete job description here...", 
        )

    analyze_button = st.button(
        "Analyze Compatibility",
        type = "primary",
    )



if analyze_button:
    if resume_file is None:
        st.error("Please upload a Resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            resume_text = extract_resume_text(resume_file)
            cleaned_resume = clean_text(resume_text)
            predicted_role = predict_job_role(
                cleaned_resume,
                model,
                tfidf
            )
            cleaned_job = clean_text(job_description)

            with upload_tab:
                st.success("Resume text extracted successfully.")

                with st.expander("Original Resume"):
                    st.text(resume_text[:2000])

                with st.expander("Cleaned Resume"):
                    st.text(cleaned_resume[:2000])

            
            match_score = calculate_match(
                cleaned_resume,
                cleaned_job
            )
            match_score = round(match_score, 1)

            matched_skills, missing_skills = compare_skills(
                cleaned_resume,
                cleaned_job
            )

            with results_tab:
                st.success("Analysis completed successfully.")

                st.subheader("Compatibility Score")
                st.metric(
                    label="Match Score",
                    value=f"{match_score}%",
                )

                st.progress(match_score / 100)

                if match_score >= 45:
                    st.success("Overall result: Good Match")
                elif match_score >= 25:
                    st.warning("Overall result: Moderate Match")
                else:
                    st.error("Overall result: Poor Match")

                st.divider()

                role_column, score_column = st.columns(2)

                with role_column:
                    st.subheader("Predicted Job Role")
                    st.info(predicted_role)

                with score_column:
                    st.subheader("Match Summary")
                    st.write(
                        f"""
                    The uploaded resume has a compatibility score of 
                    **{match_score}%** with the provided job description. 

                    The predicted job role is **{predicted_role}**.

                    {len(matched_skills)} required skills were found in the resume, 
                    while {len(missing_skills)} requiredskills were not detected. 
                    """ 
                    )

                st.divider()

                matched_column, missing_column = st.columns(2)

                with matched_column:
                    st.subheader(f"Matched Skills ({len(matched_skills)})")

                    for skill in matched_skills:
                        st.write(f"✅ {format_skill_name(skill)}")

                with missing_column:
                    st.subheader(f"Missing Skills ({len(missing_skills)})")

                    for skill in missing_skills:
                        st.write(f"❌ {format_skill_name(skill)}")

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Unable to read the resume: {error}")

